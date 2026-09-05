from typing import Any, Dict, List
import streamlit as st

from backend.core import run_llm

# --- This script is for creating chat UI using streamlit.  ---

def _format_sources(context_docs: List[Any]) -> List[str]:
    return [
        str((meta.get("source") or "Unknown"))
        for doc in (context_docs or [])
        if (meta := (getattr(doc, "metadata", None) or {})) is not None
    ]



st.set_page_config(page_title="LangChain Documentation Helper", layout="centered")
st.title("LangChain Documentation Helper")

# Display sidebar with clear chat button.
with st.sidebar:
    st.subheader("Session")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.pop("messages", None) # We saved the messages history in streamlit session state, so if the user clicks clear chat, just pop the messages to remove it.
        st.rerun() # Rerun it after removing history.


# However, when we were going to be starting the application,
# we still don't have any messages.
# So let's start by writing a placeholder message.
# So if there aren't messages in the session state,
# so this means we fired up the application,
# nothing happened yet, then we want to add
# to the messages key here in the session_state,
# we want to add here an example message.
# So let's go and add an example message
# with the role of "assistant" and the content of "Ask me anything about LangChain docs.
# I'll retrieve relevant context and cite sources."
# And let's go and write sources equals to an empty list here.
# So this is an artificial message we are going to be storing in the session_state in the messages key.
# I remind you the message_state is simply dictionary, which is going to have the key of "messages."
# And right now we are populating it with a list that contains one message.
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant", # it can be either user, assistant, ai, human, or string.
            "content": "Ask me anything about LangChain docs. I’ll retrieve relevant context and cite sources.",
            "sources": [],
        }
    ]



# And now we want to go and we want to iterate over all the messages.
# So we are going to be iterating over the session_state in the messages.
# So this is going to be a list of messages, whether a user message or an AI message.
# And here we are going to be iterating on those messages.
# So for each message we want now to create a container which is going to be holding the message.
# So let's use the with streamlit.chat_message.
# And this is here is going to insert a chat message container.
# And in its parameters we can give it a name.
# So it can be either user, assistant, ai, human, or string.
# And this is going to give that message a role,
# and it's going to have a different theme
# for the message accordingly, with an avatar,
# and we're going to be seeing it now when we run it.
# So for every message I want to go and I want to display its content.
# So everything under this indentation over here,
# under chat_message, is going to be displayed according to the current role.
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")




prompt = st.chat_input("Ask a question about LangChain…")


# this is going to be executed only if the user pressed Enter or submitted the the input.
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        try:
            with st.spinner("Retrieving docs and generating answer..."):
                result: Dict[str, Any] = run_llm(prompt)
                answer = str(result.get("answer", "")).strip() or "(No answer returned.)"
                sources = _format_sources(result.get("context", []))

            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for s in sources:
                        st.markdown(f"- {s}")

            st.session_state.messages.append(
                {"role": "assistant", "content": answer, "sources": sources}
            )
        except Exception as e:
            st.error("Failed to generate a response.")
            st.exception(e)



# def main():
#     print("Hello from langchain-course!")


# if __name__ == "__main__":
#     main()
