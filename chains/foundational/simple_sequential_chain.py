from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from loguru import logger


def main():
    load_dotenv(find_dotenv())

    # This is an LLMChain to write a synopsis given a title of a play.
    llm = OpenAI(temperature=0.0, verbose=True)
    template = """You are a playwright. Given the title of play, it is your job to write a synopsis for that title.

    Title: {title}
    Playwright: This is a synopsis for the above play:"""
    prompt = PromptTemplate.from_template(template)
    synopsis_chain = LLMChain(llm=llm, prompt=prompt)

    # This is an LLMChain to write a review of a play given a synopsis.
    llm = OpenAI(temperature=0.0)
    template = """You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.

    Play Synopsis:
    {synopsis}
    Review from a New York Times play critic of the above play:"""
    prompt = PromptTemplate.from_template(template)
    review_chain = LLMChain(llm=llm, prompt=prompt)

    # This is the overall chain where we run these two chains in sequence.
    overall_chain = SimpleSequentialChain(chains=[synopsis_chain, review_chain], verbose=True)
    review = overall_chain.run("Tragedy at sunset on the beach")
    logger.info("review: {}", review)


if __name__ == '__main__':
    main()
