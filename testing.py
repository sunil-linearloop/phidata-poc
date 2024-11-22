import nest_asyncio
import time
from typing import Optional, List, Dict
import streamlit as st
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import RatelimitException
from phi.tools.newspaper4k import Newspaper4k
from phi.utils.log import logger

from assistants import get_article_summarizer, get_article_writer  # type: ignore

nest_asyncio.apply()
st.set_page_config(
    page_title="News Articles",
    page_icon=":orange_heart:",
)
st.title("News Articles powered by Groq")
st.markdown("##### :orange_heart: built using [phidata](https://github.com/phidatahq/phidata)")

# Define use cases and their configurations
USE_CASES = {
    "Business News": {
        "default_topic": "Hashicorp IBM",
        "search_results": 7,
        "summary_length": 800,
        "draft_length": 5000,
        "prompt_placeholder": "Enter a business topic (e.g., Company mergers, Market trends)",
    },
    "Travel Guides": {
        "default_topic": "Paris Tourism",
        "search_results": 10,
        "summary_length": 1000,
        "draft_length": 6000,
        "prompt_placeholder": "Enter a destination or travel topic",
    },
    "Research Papers": {
        "default_topic": "Machine Learning Applications",
        "search_results": 15,
        "summary_length": 1200,
        "draft_length": 8000,
        "prompt_placeholder": "Enter a research topic or field",
    },
    "Technology Trends": {
        "default_topic": "AI Developments",
        "search_results": 12,
        "summary_length": 1000,
        "draft_length": 6000,
        "prompt_placeholder": "Enter a technology trend or topic",
    },
}

def truncate_text(text: str, words: int) -> str:
    return " ".join(text.split()[:words])

def get_news_with_retry(
    keywords: str, 
    max_results: int = 10, 
    max_retries: int = 3, 
    initial_delay: float = 5.0
) -> List[Dict]:
    """
    Fetch news results with exponential backoff and retry mechanism
    
    Args:
        keywords (str): Search keywords
        max_results (int): Maximum number of results to fetch
        max_retries (int): Number of retry attempts
        initial_delay (float): Initial delay between retries
    
    Returns:
        List[Dict]: List of news articles
    """
    for attempt in range(max_retries):
        try:
            with DDGS() as ddgs:
                results = ddgs.news(keywords=keywords, max_results=max_results)
                return list(results)
        except (RatelimitException, Exception) as e:
            if attempt < max_retries - 1:
                delay = initial_delay * (2 ** attempt)
                st.warning(f"Rate limit hit. Waiting {delay:.2f} seconds before retry (Attempt {attempt + 1})")
                time.sleep(delay)
            else:
                st.error(f"Failed to fetch news after {max_retries} attempts. Error: {str(e)}")
                return []
    return []

def main() -> None:
    # Select use case
    use_case = st.sidebar.selectbox(
        "Select Use Case",
        options=list(USE_CASES.keys()),
        key="use_case"
    )
    
    # Get use case configuration
    case_config = USE_CASES[use_case]
    
    # Get models
    summary_model = st.sidebar.selectbox(
        "Select Summary Model", 
        options=["llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192"]
    )
    # Set assistant_type in session state
    if "summary_model" not in st.session_state:
        st.session_state["summary_model"] = summary_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["summary_model"] != summary_model:
        st.session_state["summary_model"] = summary_model
        st.rerun()

    writer_model = st.sidebar.selectbox(
        "Select Writer Model", 
        options=["llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]
    )
    # Set assistant_type in session state
    if "writer_model" not in st.session_state:
        st.session_state["writer_model"] = writer_model
    # Restart the assistant if assistant_type has changed
    elif st.session_state["writer_model"] != writer_model:
        st.session_state["writer_model"] = writer_model
        st.rerun()

    # Checkboxes for research options
    st.sidebar.markdown("## Research Options")
    num_search_results = st.sidebar.slider(
        ":sparkles: Number of Search Results",
        min_value=3,
        max_value=20,
        value=case_config["search_results"],
        help="Number of results to search for, note only the articles that can be read will be summarized.",
    )
    per_article_summary_length = st.sidebar.slider(
        ":sparkles: Length of Article Summaries",
        min_value=100,
        max_value=2000,
        value=case_config["summary_length"],
        step=100,
        help="Number of words per article summary",
    )
    news_summary_length = st.sidebar.slider(
        ":sparkles: Length of Draft",
        min_value=1000,
        max_value=10000,
        value=case_config["draft_length"],
        step=100,
        help="Number of words in the draft article, this should fit the context length of the model.",
    )

    # Get topic for report
    article_topic = st.text_input(
        ":spiral_calendar_pad: Enter a topic",
        value=case_config["default_topic"],
        placeholder=case_config["prompt_placeholder"]
    )
    write_article = st.button("Write Article")
    
    if write_article:
        news_results = []
        news_summary: Optional[str] = None
        with st.status("Reading News", expanded=False) as status:
            with st.container():
                news_container = st.empty()
                newspaper_tools = Newspaper4k()
                
                results = get_news_with_retry(keywords=article_topic, max_results=num_search_results)
                
                for r in results:
                    if "url" in r:
                        try:
                            article_data = newspaper_tools.get_article_data(r["url"])
                            if article_data and "text" in article_data:
                                r["text"] = article_data["text"]
                                news_results.append(r)
                                if news_results:
                                    news_container.write(news_results)
                        except Exception as e:
                            st.warning(f"Could not process article {r.get('url', 'Unknown URL')}: {str(e)}")
            
            if news_results:
                news_container.write(news_results)
            status.update(label="News Search Complete", state="complete", expanded=False)

        if len(news_results) > 0:
            news_summary = ""
            with st.status("Summarizing News", expanded=False) as status:
                article_summarizer = get_article_summarizer(model=summary_model, length=per_article_summary_length)
                with st.container():
                    summary_container = st.empty()
                    for news_result in news_results:
                        news_summary += f"### {news_result['title']}\n\n"
                        news_summary += f"- Date: {news_result['date']}\n\n"
                        news_summary += f"- URL: {news_result['url']}\n\n"
                        news_summary += f"#### Introduction\n\n{news_result['body']}\n\n"

                        _summary: str = article_summarizer.run(news_result["text"], stream=False)
                        _summary_length = len(_summary.split())
                        if _summary_length > news_summary_length:
                            _summary = truncate_text(_summary, news_summary_length)
                            logger.info(f"Truncated summary for {news_result['title']} to {news_summary_length} words.")
                        news_summary += "#### Summary\n\n"
                        news_summary += _summary
                        news_summary += "\n\n---\n\n"
                        if news_summary:
                            summary_container.markdown(news_summary)
                        if len(news_summary.split()) > news_summary_length:
                            logger.info(f"Stopping news summary at length: {len(news_summary.split())}")
                            break
                if news_summary:
                    summary_container.markdown(news_summary)
                status.update(label="News Summarization Complete", state="complete", expanded=False)

        if news_summary is None:
            st.write("Sorry could not find any news or web search results. Please try again.")
            return

        article_draft = ""
        article_draft += f"# {use_case}: {article_topic}\n\n"
        if news_summary:
            article_draft += f"## Summary of Articles on {article_topic}\n\n"
            article_draft += f"This section provides a comprehensive {use_case.lower()} summary about {article_topic}.\n\n"
            article_draft += "<news_summary>\n\n"
            article_draft += f"{news_summary}\n\n"
            article_draft += "</news_summary>\n\n"

        with st.status("Writing Draft", expanded=True) as status:
            with st.container():
                draft_container = st.empty()
                draft_container.markdown(article_draft)
            status.update(label="Draft Complete", state="complete", expanded=False)

        article_writer = get_article_writer(model=writer_model)
        with st.spinner("Writing Article..."):
            final_report = ""
            final_report_container = st.empty()
            for delta in article_writer.run(article_draft):
                final_report += delta  # type: ignore
                final_report_container.markdown(final_report)

    st.sidebar.markdown("---")
    if st.sidebar.button("Restart"):
        st.rerun()

main()