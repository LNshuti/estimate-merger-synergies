**Estimate Merger Synergies -- UBS and Credit Suisse**
------------------------------------------------------
**Prompt:** You are a partner at Deloitte in Mergers and Acquisitions. You have just landed a massive project where UBS is authorized by Swiss Authorities to purchase Credit Suisse. Create a report using the latest data available about both companies using publicly available information to present to your manager the best way UBS can maximize shareholder value both in the short and long term. 

**Subject: UBS Acquisition of Credit Suisse - Maximizing Shareholder Value**

**Dear [Manager's Name],**

I am pleased to provide you with a detailed analysis of UBS's proposed acquisition of Credit Suisse. Based on the latest publicly available data, I have compiled key findings and recommendations to help UBS maximize shareholder value in the short and long term.

**Executive Summary:**
UBS Group AG, a leading global financial services company, has been authorized by Swiss Authorities to acquire Credit Suisse Group AG, another major Swiss financial institution. This acquisition presents a unique opportunity for UBS to expand its market share, enhance its product offerings, and achieve significant cost synergies. Our objective is to provide strategic guidance on how UBS can best capitalize on this opportunity to maximize shareholder value.

**Company Overview**
---------------------
**2.1 UBS Group AG (as of 31-12-2022):**
* Total assets: USD 1.1 trillion
* Market capitalization: USD 57.8 billion
* Revenue: USD 34.56 billion
* Net income: USD 7.52 billion
* Employees: ~73,000

**Core business segments: Wealth Management, Investment Banking, Asset Management, and Retail Banking**

**2.2 Credit Suisse Group AG (as of 2021):**
* Total assets: CHF 531 billion
* Market capitalization: CHF 11.1 billion
* Revenue: CHF 14.9 billion
**Net income: CHF -7.3 billion loss (due to significant legal settlements and losses)**
* Employees: ~49,000

**Core business segments: Wealth Management, Investment Banking, Asset Management, and Retail Banking**

**Notes:GPT4 has a date cutoff of 2021. Used 2022 annual reports from ubs.com and cs.com**

# What is the cutoff date for GPT-4? 

**Notes: GPT4 has a date cutoff of . We updated financial data using the latest annual reports from UBS and Credit Suisse's websites. **

**3. Strategic Recommendations**
---------------------------------
**3.1 Short-term value maximization:**

**a) Cost Synergies:** Post-acquisition, UBS should focus on achieving cost synergies by consolidating overlapping operations, eliminating duplicate positions, and streamlining processes. This will help UBS realize substantial cost savings, which can be used to improve operational efficiency and strengthen its competitive position.

**b) Debt Management:** UBS should carefully assess Credit Suisse's debt structure and identify opportunities to reduce debt-related costs. This may include refinancing high-interest debt, negotiating better terms, or restructuring debt to optimize the balance sheet.

**c) Divestiture of Non-core Assets:** UBS should consider divesting non-core assets and business units that do not align with the combined company's strategic objectives. This will generate cash and help UBS focus on its core competencies and growth areas.

**3.2 Long-term value maximization:**

**a) Strengthening Wealth Management:** The combined entity will have a significantly larger client base, enabling UBS to strengthen its Wealth Management business. UBS should focus on enhancing its product offerings, advisory services, and digital platforms to attract and retain high-net-worth clients.

**b) Expansion of Investment Banking and Asset Management:** The acquisition will provide UBS with a broader range of expertise and resources in these areas. UBS should leverage this advantage to expand its investment banking and asset management services, targeting new markets and industry sectors.

**c) Innovation and Digital Transformation:** UBS should continue to invest in innovation and digital transformation initiatives to stay competitive in an increasingly technology-driven financial services landscape. This includes enhancing its digital banking platforms, adopting artificial intelligence and machine learning tools, and exploring new growth areas like blockchain and fintech.

**Conclusion**
---------------
The acquisition of Credit Suisse presents UBS with a significant opportunity to maximize shareholder value in the short and long term. UBS can strengthen its financial position by focusing on cost synergies, debt management, and divestiture of non-core assets in the short term. In the long term, UBS should capitalize on the expanded client base and resources to further develop its core business segments, particularly in wealth management, investment banking, and asset management. Additionally, embracing innovation and digital transformation initiatives will ensure UBS remains competitive in the evolving financial services landscape.

We recommend UBS to move forward with the acquisition, keeping in mind the strategic recommendations outlined in this report. Our team at Deloitte is prepared to assist UBS throughout the acquisition process, providing expert guidance on valuation, due diligence, integration, and post-merger optimization to ensure a seamless and successful transaction.

Please feel free to reach out if you require any additional information or clarification on the findings and recommendations presented in this report. We look forward to working together to achieve the best possible outcome for UBS and its shareholders.
Sincerely,

[Your Name]

Partner, Mergers and Acquisitions

Deloitte

## Semantic Search with 2022 Annua Reports

```{python}
from langchain.embeddings.openai import OpenAIEmbeddings 
from langchain.text_splitter import CharacterTextSplitter 
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS 
from langchain import OpenAI, ConversationChain
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import numpy as np
```


```{python}
import os 

PINECONE_API_ENV = 'us-central1-gcp'
```


```{python}
ubs_reader = UnstructuredPDFLoader("/content/drive/MyDrive/ubs_cs_merger/full-report-ubs-group-ag-consolidated-4q22.pdf")
ubs_data = ubs_reader.load()
```

```{python}
creditsuisse_reader = UnstructuredPDFLoader("/content/drive/MyDrive/ubs_cs_merger/csg-ar22-compensation-en.pdf")
creditsuisse_data = creditsuisse_reader.load()
```

```{python}
llm = OpenAI(temperature=0, openai_api_key="openai_api-key")
conversation = ConversationChain(llm=llm, verbose=True)
```


```{python}
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(creditsuisse_data)
print (f'Now you have {len(texts)} documents')

texts_array = np.array(texts)
```

```{python}
import pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
```

```{python}
embedding_dim = texts_array.shape[0]  # Assuming 'embeddings' is a NumPy array or a similar data structure
print(f"Embedding dimension: {embedding_dim}")
```

```{python}
embedding_dim = texts_array.shape[0]  # Assuming 'embeddings' is a NumPy array or a similar data structure
print(f"Embedding dimension: {embedding_dim}")
```

```{python}
import pinecone
#pinecone.delete_index("mergers-and-acqs")
pinecone.create_index("mergers-and-acqs", dimension=1536)
index_description = pinecone.describe_index('mergers-and-acqs')
```

```{python}
docsearch = Pinecone.from_texts([t.page_content for t in texts_array], embeddings, index_name='mergers-and-acqs')
query = "What were the total assets?  List the date on the docsearch"
docs = docsearch.similarity_search(query, include_metadata=True)
```
