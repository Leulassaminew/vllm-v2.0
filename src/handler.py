import runpod
from utils import JobInput
from engine import vLLMEngine
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset
import pandas as pd

dataset=load_dataset("meetplace1/clientdata.csv")
df = pd.DataFrame(dataset['train'])
sentences = df["Question"].values
vectorizer = TfidfVectorizer()
vectorizer.fit(sentences)
sentence_vectors = vectorizer.transform(sentences)
vllm_engine = vLLMEngine()

def context(job_input):
    messages=job_input["messages"][-1]
    messages=messages["content"]
    text_vector = vectorizer.transform([messages])
    similarity_scores = cosine_similarity(text_vector, sentence_vectors)
    most_similar_index = similarity_scores.argmax()
    top_indices = similarity_scores.argsort()[0][-2:]
    top=top_indices[::-1]
    top_sentences = sentences[top]
    answer = ''
    if similarity_scores[0][most_similar_index] >= 0.4:
        for sent in top_sentences:
            res = df[df["Question"] == sent]["Answer"].values[0]
            answer += res
            # txt+=res
            answer = answer.replace("•", " ")
            answer = answer.replace("\n", "")
    else:
        answer = ''
    return answer

async def handler(job):
    cont=context(job["input"])
    job["input"]["context"]=cont
    job_input = JobInput(job["input"])
    results_generator = vllm_engine.generate(job_input)
    async for batch in results_generator:
        yield batch
        
runpod.serverless.start(
    {
        "handler": handler,
        "concurrency_modifier": lambda x: vllm_engine.max_concurrency,
        "return_aggregate_stream": True,
    }
)
