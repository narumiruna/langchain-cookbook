from dotenv import find_dotenv, load_dotenv
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from loguru import logger


def main():
    """https://python.langchain.com/docs/modules/chains/foundational/sequential_chains"""
    load_dotenv(find_dotenv())

    llm = OpenAI(temperature=0.0, verbose=True)
    template = """
    Food: {food}
    Ingredients:
    """
    prompt = PromptTemplate.from_template(template)
    ingredient_chain = LLMChain(llm=llm, prompt=prompt)

    llm = OpenAI(temperature=0.0)
    template = """
    Ingredients: {ingredients}
    Instructions:
    """
    prompt = PromptTemplate.from_template(template)
    instruction_chain = LLMChain(llm=llm, prompt=prompt)

    recipe_chain = SimpleSequentialChain(chains=[ingredient_chain, instruction_chain], verbose=True)
    review = recipe_chain.run("vanilla ice cream")
    logger.info("recipe: {}", review)


if __name__ == '__main__':
    main()
