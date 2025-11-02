"""
Streamlit Dashboard for Chat Summarization
Interactive demo interface for testing the summarization model
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Page configuration
st.set_page_config(
    page_title="Chat Summarization Demo",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .summary-box {
        background-color: #e3f2fd;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
    .example-box {
        background-color: #f5f5f5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        cursor: pointer;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load model and tokenizer (cached)"""
    model_path = "models/best_model"
    
    with st.spinner("Loading AI model... This may take a moment."):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.eval()
        
        # Move to appropriate device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    
    return tokenizer, model

def generate_summary(conversation, tokenizer, model, max_length=128, num_beams=4):
    """Generate summary for a conversation"""
    # Tokenize input
    inputs = tokenizer(
        conversation,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to same device as model
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Generate summary
    with torch.no_grad():
        summary_ids = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=2.0,
            early_stopping=True
        )
    
    # Decode summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Example conversations
EXAMPLES = {
    "Handoff - Laptop Not Turning On": """Agent Sarah: Hello! Thank you for contacting TechSupport. How can I help you today?
Customer: Hi, my laptop isn't turning on at all. I've tried pressing the power button multiple times.
Agent Sarah: I'm sorry to hear that. When did this issue start?
Customer: This morning. It was working fine yesterday evening.
Agent Sarah: Okay. Have you tried plugging it into a different power outlet?
Customer: Yes, I tried three different outlets. Still nothing.
Agent Sarah: Let me check your warranty status. Can you provide your serial number?
Customer: Sure, it's SN-12345-XYZ.
Agent Sarah: Thank you. Your device is still under warranty. I'm creating a replacement order for you.
Customer: Oh great! How long will that take?
Agent Sarah: You should receive it within 3-5 business days. I'm going to transfer you to my colleague Tom who will finalize the shipping details and tracking information.""",
    
    "Handoff - Delayed Order Refund": """Agent Mike: Good afternoon! How may I assist you today?
Customer: I ordered a dress two weeks ago and it still hasn't arrived. Order number ORD-789456.
Agent Mike: I apologize for the inconvenience. Let me look into this for you... I see your order was shipped on the 15th but it's delayed at the distribution center.
Customer: That's frustrating. I needed it for an event this weekend which is tomorrow.
Agent Mike: I completely understand. Since it's delayed and time-sensitive, I can offer you a full refund or rush ship a replacement overnight.
Customer: I'll take the refund please. There's no time for replacement.
Agent Mike: Absolutely. I've initiated the refund process. Let me transfer you to our billing specialist Jessica who will confirm the refund amount and timeline.""",
    
    "Handoff - Account Access Issue": """Agent Lisa: Thank you for calling customer support. What can I help you with?
Customer: I can't log into my account. It keeps saying my password is incorrect but I'm 100% sure it's right.
Agent Lisa: I understand how frustrating that must be. Let me pull up your account. Can you provide your email address?
Customer: It's john.smith@email.com
Agent Lisa: Thank you. I see there were 5 failed login attempts in the past hour, which triggered our security protocol and temporarily locked your account.
Customer: Oh no! I have an important meeting in 10 minutes and need access to my files.
Agent Lisa: I can unlock your account right now. For security, I need to verify two pieces of information. What's your date of birth?
Customer: March 15, 1985
Agent Lisa: And the last 4 digits of the phone number on file?
Customer: 7392
Agent Lisa: Perfect, I've verified your identity and unlocked the account. However, you'll need to reset your password. I'm transferring you to our technical team member David who will walk you through the password reset process and ensure you can access your files before your meeting.""",
    
    "Handoff - Billing Dispute": """Agent Maria: Hello, this is Maria from billing support. How can I help?
Customer: Hi, I was charged twice for my subscription this month. I see two charges of $29.99 on my credit card.
Agent Maria: I'm so sorry about that! Let me investigate immediately. Can you provide your account number?
Customer: It's ACC-456789
Agent Maria: Thank you. I'm looking at your billing history now... I can confirm there was a duplicate charge on November 1st due to a system error.
Customer: Okay, so when can I expect the refund?
Agent Maria: I've flagged this for immediate refund processing. The duplicate charge of $29.99 will be refunded to your original payment method. Let me transfer you to our refund specialist Carlos who will provide you with a refund confirmation number and exact timeline.""",
    
    "Handoff - Product Return": """Agent Kevin: Good morning! Thanks for contacting us. What brings you in today?
Customer: I bought a pair of headphones last week but they're not working properly. The left ear cuts out constantly.
Agent Kevin: I'm sorry the headphones aren't working as expected. Do you have your order number handy?
Customer: Yes, it's ORD-334455
Agent Kevin: Thanks! I see you purchased the WirelessPro headphones on October 28th. They're definitely still within our 30-day return window.
Customer: Great. What's the return process?
Agent Kevin: Since the product is defective, we can either send you a replacement or process a full refund. Which would you prefer?
Customer: I'd like a replacement please. These were a gift and I really want them to work.
Agent Kevin: Absolutely! I'm going to transfer you to our returns specialist Amanda who will generate your return label and expedite the replacement shipment so you get your new headphones as quickly as possible."""
}

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">💬 Chat Summarization AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Instantly summarize customer support conversations using AI</p>', unsafe_allow_html=True)
    
    # Load model
    try:
        tokenizer, model = load_model()
        st.success("✅ AI Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        max_length = st.slider(
            "Max Summary Length",
            min_value=50,
            max_value=200,
            value=128,
            step=10,
            help="Maximum number of tokens in the generated summary"
        )
        
        num_beams = st.slider(
            "Beam Search Width",
            min_value=1,
            max_value=10,
            value=4,
            help="Higher values may produce better summaries but take longer"
        )
        
        st.divider()
        
        st.header("📊 Model Info")
        st.info("""
        **Model:** BART-base fine-tuned on SAMSum
        
        **Performance:**
        - ROUGE-1: 0.477
        - ROUGE-2: 0.246
        - ROUGE-L: 0.403
        
        **Training:** 11,784 conversations
        """)
        
        st.divider()
        
        st.header("💡 About")
        st.write("""
        This AI system automatically generates concise summaries of customer support conversations, 
        helping agents quickly understand context during handoffs.
        
        **Use Case:** Reduce agent onboarding time from 2-3 minutes to seconds!
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📝 Input Conversation")
        
        # Example selector
        st.subheader("Try an Example:")
        example_choice = st.selectbox(
            "Select an example conversation",
            [""] + list(EXAMPLES.keys()),
            label_visibility="collapsed"
        )
        
        # Text input
        if example_choice:
            default_text = EXAMPLES[example_choice]
        else:
            default_text = "Paste or type a conversation here...\n\nFormat:\nPerson1: Message\nPerson2: Message"
        
        conversation = st.text_area(
            "Conversation",
            value=default_text,
            height=400,
            label_visibility="collapsed"
        )
        
        # Generate button
        generate_button = st.button("🚀 Generate Summary", type="primary", use_container_width=True)
    
    with col2:
        st.header("✨ Generated Summary")
        
        if generate_button:
            if len(conversation.strip()) < 10:
                st.warning("⚠️ Please enter a conversation with at least 10 characters.")
            else:
                with st.spinner("🤖 AI is analyzing the conversation..."):
                    try:
                        # Generate summary
                        summary = generate_summary(
                            conversation, 
                            tokenizer, 
                            model,
                            max_length=max_length,
                            num_beams=num_beams
                        )
                        
                        # Calculate metrics
                        conv_words = len(conversation.split())
                        summary_words = len(summary.split())
                        compression_ratio = conv_words / summary_words if summary_words > 0 else 0
                        
                        # Display summary
                        st.markdown(f"""
                        <div class="summary-box">
                            <h3>📋 Summary</h3>
                            <p style="font-size: 1.1rem; line-height: 1.6;">{summary}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display metrics
                        st.subheader("📊 Metrics")
                        
                        metric_col1, metric_col2, metric_col3 = st.columns(3)
                        
                        with metric_col1:
                            st.metric(
                                label="Original Length",
                                value=f"{conv_words} words"
                            )
                        
                        with metric_col2:
                            st.metric(
                                label="Summary Length",
                                value=f"{summary_words} words"
                            )
                        
                        with metric_col3:
                            st.metric(
                                label="Compression",
                                value=f"{compression_ratio:.1f}x",
                                delta=f"-{((1 - 1/compression_ratio) * 100):.0f}%"
                            )
                        
                        # Time saved estimate
                        time_saved = (conv_words - summary_words) * 0.25  # ~0.25 seconds per word
                        st.success(f"⏱️ Estimated time saved: **{time_saved:.1f} seconds** per handoff")
                        
                    except Exception as e:
                        st.error(f"❌ Error generating summary: {e}")
        else:
            st.info("👈 Enter a conversation and click 'Generate Summary' to see the AI in action!")
            
            # Show example output
            st.markdown("---")
            st.subheader("Example Output:")
            st.markdown("""
            <div class="summary-box">
                <h3>📋 Summary</h3>
                <p style="font-size: 1.1rem; line-height: 1.6;">
                Customer's laptop isn't turning on. Agent verified troubleshooting steps and found device is under warranty. 
                Replacement order created with 3-5 business day delivery.
                </p>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()