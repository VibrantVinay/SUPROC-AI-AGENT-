import streamlit as st
import json
from agent.agent import run_agent
from agent.llm_client import LLMClient

# --- Page Configuration ---
st.set_page_config(
    page_title="Suproc | AI Procurement Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for a professional look ---
st.markdown("""
    <style>
    .stButton>button { width: 100%; font-weight: bold; }
    .status-passed { color: #00C851; font-weight: bold; }
    .status-failed { color: #ff4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Agent Settings")
    model_name = st.text_input("Ollama Model", value="qwen3:4b")
    num_results = st.number_input("Requested Matches", min_value=1, max_value=10, value=3)
    st.markdown("---")
    st.info("The agent will automatically fall back to deterministic parsing if Ollama is unreachable.")

# --- Main UI ---
st.title("💼 Suproc AI Agent")
st.markdown("Enter your business requirement below to discover, verify, and draft outreach to the best suppliers.")

query = st.text_area(
    "Business Requirement", 
    placeholder="e.g., We are a startup based in Bengaluru. We need three suppliers from South India that provide food-grade biodegradable containers...",
    height=100
)

if st.button(" Run Analysis", type="primary"):
    if not query.strip():
        st.warning("Please enter a business requirement to proceed.")
    else:
        with st.spinner("Analyzing requirements, searching local dataset, and validating matches..."):
            
            # Initialize Client and Run Agent
            client = LLMClient(model=model_name)
            result = run_agent(query, requested_results=num_results, llm_client=client)
            
            st.toast("Analysis Complete!", icon="✅")
            st.divider()

            # --- Layout: Tabs for organized data presentation ---
            tab_overview, tab_matches, tab_plan, tab_raw = st.tabs([
                " Overview & Outreach", 
                " Recommended Matches", 
                " Execution & Validation", 
                " Raw JSON"
            ])

            # TAB 1: OVERVIEW & OUTREACH
            with tab_overview:
                col1, col2, col3 = st.columns(3)
                col1.metric("Objective", result.interpreted_requirement.entity_type.title())
                col2.metric("Requested", result.interpreted_requirement.requested_results)
                col3.metric("Found", len(result.recommended_matches))

                st.subheader("📝 Action Required")
                st.info(result.recommended_next_action)
                
                if result.draft_outreach_message:
                    st.text_area("Draft Outreach Message (Ready for Review)", result.draft_outreach_message, height=250)
                
                if result.human_approval_required:
                    if st.button("✅ Approve & Execute Action", type="primary"):
                        st.success("Action Approved! (Note: In this demo, no actual emails are sent).")

                if result.missing_information or result.risks_or_uncertainties:
                    st.markdown("### ⚠️ Notices & Risks")
                    for m in result.missing_information:
                        st.warning(f"**Missing Info:** {m}")
                    for r in result.risks_or_uncertainties:
                        st.error(f"**Risk:** {r}")

            # TAB 2: RECOMMENDED MATCHES
            with tab_matches:
                if not result.recommended_matches:
                    st.error("No valid matches found satisfying all hard constraints.")
                else:
                    for idx, match in enumerate(result.recommended_matches):
                        with st.expander(f"#{idx+1}: {match.name} ({match.entity_id}) - Score: {match.score.total}", expanded=(idx==0)):
                            mc1, mc2, mc3 = st.columns(3)
                            mc1.metric("Relevance", match.score.product_or_skill_relevance)
                            mc2.metric("Constraint Compliance", match.score.hard_constraint_compliance)
                            mc3.metric("Reputation", match.score.reputation)
                            
                            st.markdown("**Evidence Trail:**")
                            for evidence in match.evidence:
                                st.markdown(f"- {evidence}")

            # TAB 3: EXECUTION PLAN & VALIDATION
            with tab_plan:
                st.subheader("Execution Plan")
                for step_num, step in enumerate(result.plan.steps, 1):
                    st.markdown(f"**{step_num}.** {step}")
                
                st.markdown("---")
                st.subheader("Validation Status")
                status_color = "status-passed" if "passed" in result.validation_status else "status-failed"
                st.markdown(f"<span class='{status_color}'>{result.validation_status.upper()}</span>", unsafe_allow_html=True)
                st.caption(f"Correction attempts used: {result.correction_attempts}")
                
                if result.notes:
                    for note in result.notes:
                        st.caption(f"📝 {note}")

            # TAB 4: RAW JSON
            with tab_raw:
                st.json(result.model_dump_json())