# encoding:utf-8

import time

import openai
import openai.error
import requests
import json  # for parsing the JSON api responses and model outputs

from bot.bot import Bot
from bot.chatgpt.chat_gpt_session import ChatGPTSession
from bot.openai.open_ai_image import OpenAIImage
from bot.session_manager import SessionManager
from bridge.context import ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from common.token_bucket import TokenBucket
from config import conf, load_config
from numpy import dot  # for cosine similarity

# OpenAI对话模型API (可用)
class ChatGPTBot(Bot, OpenAIImage):

    def __init__(self):
        super().__init__()
        # set the default api_key
        openai.api_key = conf().get("open_ai_api_key")
        if conf().get("open_ai_api_base"):
            openai.api_base = conf().get("open_ai_api_base")
        proxy = conf().get("proxy")
        if proxy:
            openai.proxy = proxy
        if conf().get("rate_limit_chatgpt"):
            self.tb4chatgpt = TokenBucket(conf().get("rate_limit_chatgpt", 20))

        self.sessions = SessionManager(ChatGPTSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.args = {
            "model": conf().get("model") or "gpt-3.5-turbo",  # 对话模型的名称
            "temperature": conf().get("temperature", 0.9),  # 值在[0,1]之间，越大表示回复越具有不确定性
            # "max_tokens":4096,  # 回复最大的字符数
            "top_p": conf().get("top_p", 1),
            "frequency_penalty": conf().get("frequency_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "presence_penalty": conf().get("presence_penalty", 0.0),  # [-2,2]之间，该值越大则更倾向于产生不同的内容
            "request_timeout": conf().get("request_timeout", None),  # 请求超时时间，openai接口默认设置为600，对于难问题一般需要较长时间
            "timeout": conf().get("request_timeout", None),  # 重试超时时间，在这个时间内，将会自动重试
        }

    def choice_agent_with_query(self, query):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[{"role": "user", "content": f"""
            请针对 >>> 和 <<< 中间的用户问题，判断是否属于无法提供最新的信息或数据，是否涉及特定的法律、科技、医学或政治领域，是否需要个性化或地域化的信息时，如果属于上述范畴，请你直接回复：<!--WEB-SEARCH-GO-->。
            >>> {query}  <<<
            """}
            ],
            temperature=0.5,
        )
        logger.info(completion.choices[0]["message"]["content"])

        return {
            "total_tokens": completion["usage"]["total_tokens"],
            "completion_tokens": completion["usage"]["completion_tokens"],
            "content": completion.choices[0]["message"]["content"],
        }

    def json_gpt(self, input: str):
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=[
                {"role": "system", "content": "Output only valid JSON"},
                {"role": "user", "content": input},
            ],
            temperature=0.5,
        )

        text = completion.choices[0].message.content
        parsed = json.loads(text)
        logger.info(parsed)

        return parsed

    def embeddings(self, input: list[str]) -> list[list[str]]:
        response = openai.Embedding.create(model="text-embedding-ada-002", input=input)
        return [data.embedding for data in response.data]

    def search_web(self,
        query: str,
        google_search_key: str = "AIzaSyAQ70Mo8f138l32jxqr3ZuDXS4jy99lA2g",
        cx: str = "f1bbdda6e1ffe40d5",
    ) -> dict:
        response = requests.get(
            "https://www.googleapis.com/customsearch/v1/siterestrict",
            params={
                "q": query,
                "key": google_search_key,
                "cx": cx,
            },
        )
        return response.json()

    def reply_search(self, query):
        QUERIES_INPUT = f"""
        You have access to a search API that returns recent news articles.
        Generate an array of search queries that are relevant to this question.
        Use a variation of related keywords for the queries, trying to be as general as possible.
        Include as many queries as you can think of, including and excluding terms.
        For example, include queries like ['keyword_1 keyword_2', 'keyword_1', 'keyword_2'].
        Be creative. The more queries you include, the more likely you are to find relevant results.
        Do not repeat between keywords.

        User question: {query}

        Format: {{"queries": ["query_1", "query_2", "query_3"]}}
        """

        queries = self.json_gpt(QUERIES_INPUT)["queries"]

        # Let's include the original question as well for good measure
        if query not in queries:
            queries.append(query)

        queries

        articles = []
        i = 0

        for query in queries:
            result = self.search_web(query)
            i = i + 1
            if (i > 3):
              break
            if result.get("error") is None:
                articles = articles + result["items"]
            else:
                raise Exception(result["error"])

        # remove duplicates
        articles = list({article["link"]: article for article in articles}.values())

        print("Total number of articles:", len(articles))

        HA_INPUT = f"""
        Generate a hypothetical answer to the user's question. This answer will be used to rank search results.
        Pretend you have all the information you need to answer, but don't use any actual facts. Instead, use placeholders
        like NAME did something, or NAME said something at PLACE.

        User question: {query}

        Format: {{"hypotheticalAnswer": "hypothetical answer text"}}
        """

        hypothetical_answer = self.json_gpt(HA_INPUT)["hypotheticalAnswer"]

        hypothetical_answer

        hypothetical_answer_embedding = self.embeddings(hypothetical_answer)[0]
        article_embeddings = self.embeddings(
            [
                f"{article['title']} {article['link']} {article['snippet'][0:100]}"
                for article in articles
            ]
        )

        # Calculate cosine similarity
        cosine_similarities = []
        for article_embedding in article_embeddings:
            cosine_similarities.append(dot(hypothetical_answer_embedding, article_embedding))

        cosine_similarities[0:10]

        scored_articles = zip(articles, cosine_similarities)

        # Sort articles by cosine similarity
        sorted_articles = sorted(scored_articles, key=lambda x: x[1], reverse=True)

        # Print top 5 articles
        print("Top 5 articles:", "\n")

        for article, score in sorted_articles[0:5]:
            print("Title:", article["title"])
            print("link:", article["link"])
            print("snippet:", article["snippet"])
            print("Score:", score)
            print()

        formatted_top_results = [
            {
                "标题": article["title"],
                "链接": article["link"],
                "摘要": article["snippet"],
            }
            for article, _score in sorted_articles[0:5]
        ]
        return formatted_top_results

    def reply(self, query, context=None):
        # acquire reply content
        if context.type == ContextType.TEXT:
            logger.info("[CHATGPT] query={}".format(query))

            session_id = context["session_id"]
            reply = None
            clear_memory_commands = conf().get("clear_memory_commands", ["#清除记忆"])
            if query in clear_memory_commands:
                self.sessions.clear_session(session_id)
                reply = Reply(ReplyType.INFO, "记忆已清除")
            elif query == "#清除所有":
                self.sessions.clear_all_session()
                reply = Reply(ReplyType.INFO, "所有人记忆已清除")
            elif query == "#更新配置":
                load_config()
                reply = Reply(ReplyType.INFO, "配置已更新")
            if reply:
                return reply
            session = self.sessions.session_query(query, session_id)
            logger.debug("[CHATGPT] session query={}".format(session.messages))

            api_key = context.get("openai_api_key")
            model = context.get("gpt_model")
            new_args = None
            if model:
                new_args = self.args.copy()
                new_args["model"] = model
            # if context.get('stream'):
            #     # reply in stream
            #     return self.reply_text_stream(query, new_query, session_id)

            choice = self.choice_agent_with_query(query)
            if ("<!--WEB-SEARCH-GO-->" in choice["content"]):
                reply_content = self.reply_search(query)
                reply = Reply(ReplyType.TEXT, reply_content)
                self.sessions[session.session_id].set_system_prompt
                return reply

            reply_content = self.reply_text(session, api_key, args=new_args)
            logger.debug(
                "[CHATGPT] new_query={}, session_id={}, reply_cont={}, completion_tokens={}".format(
                    session.messages,
                    session_id,
                    reply_content["content"],
                    reply_content["completion_tokens"],
                )
            )
            if reply_content["completion_tokens"] == 0 and len(reply_content["content"]) > 0:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
            elif reply_content["completion_tokens"] > 0:
                self.sessions.session_reply(reply_content["content"], session_id, reply_content["total_tokens"])
                reply = Reply(ReplyType.TEXT, reply_content["content"])
            else:
                reply = Reply(ReplyType.ERROR, reply_content["content"])
                logger.debug("[CHATGPT] reply {} used 0 tokens.".format(reply_content))
            return reply

        elif context.type == ContextType.IMAGE_CREATE:
            ok, retstring = self.create_img(query, 0)
            reply = None
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, retstring)
            else:
                reply = Reply(ReplyType.ERROR, retstring)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply

    def reply_text(self, session: ChatGPTSession, api_key=None, args=None, retry_count=0) -> dict:
        """
        call openai's ChatCompletion to get the answer
        :param session: a conversation session
        :param session_id: session id
        :param retry_count: retry count
        :return: {}
        """
        try:
            if conf().get("rate_limit_chatgpt") and not self.tb4chatgpt.get_token():
                raise openai.error.RateLimitError("RateLimitError: rate limit exceeded")
            # if api_key == None, the default openai.api_key will be used
            if args is None:
                args = self.args
            response = openai.ChatCompletion.create(api_key=api_key, messages=session.messages, **args)
            # logger.debug("[CHATGPT] response={}".format(response))
            # logger.info("[ChatGPT] reply={}, total_tokens={}".format(response.choices[0]['message']['content'], response["usage"]["total_tokens"]))
            return {
                "total_tokens": response["usage"]["total_tokens"],
                "completion_tokens": response["usage"]["completion_tokens"],
                "content": response.choices[0]["message"]["content"],
            }
        except Exception as e:
            need_retry = retry_count < 2
            result = {"completion_tokens": 0, "content": "我现在有点累了，等会再来吧"}
            if isinstance(e, openai.error.RateLimitError):
                logger.warn("[CHATGPT] RateLimitError: {}".format(e))
                result["content"] = "提问太快啦，请休息一下再问我吧"
                if need_retry:
                    time.sleep(20)
            elif isinstance(e, openai.error.Timeout):
                logger.warn("[CHATGPT] Timeout: {}".format(e))
                result["content"] = "我没有收到你的消息"
                if need_retry:
                    time.sleep(5)
            elif isinstance(e, openai.error.APIError):
                logger.warn("[CHATGPT] Bad Gateway: {}".format(e))
                result["content"] = "请再问我一次"
                if need_retry:
                    time.sleep(10)
            elif isinstance(e, openai.error.APIConnectionError):
                logger.warn("[CHATGPT] APIConnectionError: {}".format(e))
                need_retry = False
                result["content"] = "我连接不到你的网络"
            else:
                logger.exception("[CHATGPT] Exception: {}".format(e))
                need_retry = False
                self.sessions.clear_session(session.session_id)

            if need_retry:
                logger.warn("[CHATGPT] 第{}次重试".format(retry_count + 1))
                return self.reply_text(session, api_key, args, retry_count + 1)
            else:
                return result


class AzureChatGPTBot(ChatGPTBot):
    def __init__(self):
        super().__init__()
        openai.api_type = "azure"
        openai.api_version = conf().get("azure_api_version", "2023-06-01-preview")
        self.args["deployment_id"] = conf().get("azure_deployment_id")

    def create_img(self, query, retry_count=0, api_key=None):
        api_version = "2022-08-03-preview"
        url = "{}dalle/text-to-image?api-version={}".format(openai.api_base, api_version)
        api_key = api_key or openai.api_key
        headers = {"api-key": api_key, "Content-Type": "application/json"}
        try:
            body = {"caption": query, "resolution": conf().get("image_create_size", "256x256")}
            submission = requests.post(url, headers=headers, json=body)
            operation_location = submission.headers["Operation-Location"]
            retry_after = submission.headers["Retry-after"]
            status = ""
            image_url = ""
            while status != "Succeeded":
                logger.info("waiting for image create..., " + status + ",retry after " + retry_after + " seconds")
                time.sleep(int(retry_after))
                response = requests.get(operation_location, headers=headers)
                status = response.json()["status"]
            image_url = response.json()["result"]["contentUrl"]
            return True, image_url
        except Exception as e:
            logger.error("create image error: {}".format(e))
            return False, "图片生成失败"
