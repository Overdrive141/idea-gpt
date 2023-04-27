import os
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate 
# LLM Chain allows us to run our topic through our prompttemplate then run it
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain

# Give memory to our LLM
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper # API Calls to Wikipedia

# SequentialChain - Allows us to see output of every chain individually

# SimpleSequentialChain - Allows to stack bunch of chains for multiple outputs. Limitation: Does not allow individual output but only last output

# LangChain - used to build llm workflow
# Wikipedia - Connect GPT to Wikipedia
# ChromaDB - Vector Storage
# TikToken - Backend Tokenizer for OpenAI

os.environ['OPENAI_API_KEY']= apikey

st.title("🦜🔗 Farhan's IdeaGPT")
prompt = st.text_input("Enter a topic you want to generate an idea about, and generate an article based on that idea after Wikipedia's research. E.g, Frontend Design")

# Templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='Generate one unique idea about {topic}'
)
script_template = PromptTemplate(
    input_variables=['title'],
    template='Write a sample article about this idea. IDEA: {title}'
)
# Script Template Part 2 With Wikipedia
wiki_script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write a sample article or blogpost about this idea. IDEA: {title} | Use this wikipedia research if you think it will improve the article: {wikipedia_research}'
)


# Memory - Using it for history, not for prompting
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


#LLMs
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=wiki_script_template, verbose=True, output_key='script', memory=script_memory)

# AutoGPT Magic Happens Here
## Removed since it does not allow indivudual outputs;
# simple_seqential_chain = SimpleSequentialChain(chains=[title_chain,script_chain], verbose=True)
# seqential_chain = SequentialChain(chains=[title_chain,script_chain], verbose=True, input_variables=['topic'],
#                                   output_variables=['title', 'script'])

# Part 3: Sequential Chains no longer used. Now I am using wiki
wiki = WikipediaAPIWrapper()

# Show response if prompt
if prompt:
    title = title_chain.run(topic=prompt)
    st.header('Idea: ')
    st.write(title)
    
    wiki_research = wiki.run(prompt)
    st.header('Research from Wiki: ')
    st.caption(wiki_research)

    script = script_chain.run(title=title, wikipedia_research=wiki_research) # takes 2 inputs
    # response = script_chain.run(title=title)

    # No longer used after adding Wiki API
    # response = seqential_chain({'topic': prompt})
    # st.write('Title: ' + response['title'])
    # st.write('Script: ' + response['script'])

    st.header('Article')
    st.write(script)

    with st.expander('Message History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wiki Research History'):
        st.info(wiki_research)