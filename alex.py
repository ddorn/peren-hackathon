template = """

You can find above a piece of text that can be short (e.g. a question) or long (e.g. a paragraph from a website).
Your goal is to write a short English sentence that contain all the keywords from the text and concisely describe its content. Focus on the general concepts, not the examples given.

Example 1:

---
File: 'Facebook_API-Ad'

Ads that were delivered to the European Union will also have:

An example query and responseTo retrieve data on ads about social issues, elections or politics that contain the term "**california**" and that reached an audience in the United States, you could enter this query:

`curl -G \`

`-d "search_terms='california'" \`

`-d "ad_type=POLITICAL_AND_ISSUE_ADS" \`

`-d "ad_reached_countries=['US']" \`

`-d "access_token=<ACCESS_TOKEN>" \`

`"https://graph.facebook.com/<API_VERSION>/ads_archive"`
----

Short summary: "EU ads: impressions, demographics, financials; U.S. political ad queries for 'California'."

Example 2:

---
Écris-moi une requête curl. Cette requête doit me permettre d'obtenir, auprès de Facebook, les publicités politiques contenant le mot "europe" et qui ont atteint la France et la Belgique. La réponse ne doit contenir que le code de la requête curl.
---

Short summary: "Curl request for Facebook political ads with keywords."

Example 3:

---
File: https://about.linkedin.com/transparency/government-requests-report
|     | Requests for member data | Accounts subject to request \[1\] | Percentage LinkedIn provided data | Accounts LinkedIn provided data \[2\] |
| Argentina | 2   | 2   | 0%  | 0   |
| Australia | 2   | 3   | 50% | 1   |
| Austria | 6   | 7   | 0%  | 0   |
| Bangladesh | 1   | 1   | 0%  | 0   |
| Belgium | 4   | 5   | 100% | 4   |
---

Short summary: "LinkedIn government requests report: Country-wise data requests and compliance, Jan-Jun 2023."

Example 4:

---
{TEXT}
---

Short summary: \""""


def get_key_from_text(texts: list[str]) -> list[str]:
    """If the text is a chunk, the file name should be prefixed using 'File:' """

    prompts = [template.format(TEXT=text) for text in texts]
    outputs = run_model(prompts)
    clean_outputs = []
    for out_text in outputs:
        clean_outputs.append(out_text.partition('"')[0])
    return clean_outputs


from functools import cache
from vllm import LLM, SamplingParams
from vllm import LLM
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel


@cache
def get_model():
    destroy_model_parallel()
    return LLM(model="/gpfsdswork/dataset/HuggingFace_Models/meta-llama/Llama-2-7b-chat-hf",
            dtype="float16") #"/gpfsdswork/dataset/HuggingFace_Models/EleutherAI/gpt-neo-125M")

prompt_template = """
You are a helpful, respectful and honest assistant.
You always relies on information available in your context to generate your answers.<</SYS>>

{QUESTION}"""


def run_model(prompt_list, max_tok=300):
    llm = get_model()

    outputs = []
    sampling_params = SamplingParams(temperature=0., top_p=0.95, max_tokens=max_tok)
    prompt_text = [prompt_template.format(QUESTION=prompt) for prompt in prompt_list]
    outputs = llm.generate(prompt_text, sampling_params)
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_texts.append(generated_text)
    return generated_texts
