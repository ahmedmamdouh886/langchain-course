from dotenv import load_dotenv
from graph.graph import app

load_dotenv()


if __name__ == "__main__":
    print("Hello Advanced RAG")

    # The router will route this to the "retrieve" RAG workflow.
    # print(app.invoke(input={"question": "what is agent memory?"}))

    # The router will route this to the "websearch" workflow.
    print(app.invoke(input={"question": "How to make pizza?"}))
