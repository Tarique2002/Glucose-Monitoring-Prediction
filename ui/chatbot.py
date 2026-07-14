import streamlit as st
import time

def render_chatbot():
    st.markdown("<h2 class='animate-fade-in'>🤖 AI Healthcare Assistant</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color: var(--text-muted);'>Ask me questions about diabetes, glucose levels, or lifestyle improvements.</p>", unsafe_allow_html=True)
    
    st.warning("⚠️ Disclaimer: This AI is for educational purposes only and does not provide medical advice. Consult a doctor for medical decisions.")
    
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I am your AI Health Assistant. How can I help you understand your metrics today?"}
        ]
        
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if prompt := st.chat_input("Ask about BMI, Glucose, or Diabetes..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Simple rule-based mock for portfolio demonstration
            prompt_lower = prompt.lower()
            if "bmi" in prompt_lower:
                reply = "BMI stands for Body Mass Index. It's calculated by dividing your weight in kilograms by the square of your height in meters. A BMI between 18.5 and 24.9 is considered healthy."
            elif "food" in prompt_lower or "diet" in prompt_lower or "avoid" in prompt_lower:
                reply = "To manage glucose levels, it's generally recommended to avoid refined carbohydrates (like white bread and sugary drinks), highly processed foods, and trans fats. Focus on whole grains, lean proteins, and leafy greens."
            elif "insulin" in prompt_lower:
                reply = "Insulin is a hormone created by your pancreas that controls the amount of glucose in your bloodstream at any given moment. It also helps store glucose in your liver, fat, and muscles."
            elif "exercise" in prompt_lower:
                reply = "Regular exercise improves insulin sensitivity, meaning your cells are better able to use available sugar in your bloodstream. Aim for at least 150 minutes of moderate aerobic activity per week."
            else:
                reply = "That's a great question! While I am currently a demonstration model, in a production environment, I would connect to a Large Language Model (like GPT-4 or Gemini) to provide a comprehensive answer based on medical literature."
                
            # Simulate stream
            for chunk in reply.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
        st.session_state.messages.append({"role": "assistant", "content": full_response})
