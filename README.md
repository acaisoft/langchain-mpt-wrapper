# langchain-mpt-wrapper
langchain-mpt-wrapper provides a wrapper of Mosaicml MPT-7b model that can be used in Langchain framework
# Installation
`pip install langchain-mpt-wrapper`
# Example usage
```
import os
from langchain import LLMChain, PromptTemplate
from langchain_mpt_wrapper import LangchainMPTWrapper


model_path = os.environ['MPT_MODEL_PATH']
llm = LangchainMPTWrapper(model_path)

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template("{text}")
)
output = chain.run("The sky is")
print(output)
```
