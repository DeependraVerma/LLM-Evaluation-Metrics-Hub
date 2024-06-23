import streamlit as st
from evaluate import LLM_Evaluator

# Initialize evaluator
evaluator = LLM_Evaluator()

# Main page title and sidebar options
main_title = st.empty()
option = st.sidebar.radio("Select Option", ("Evaluate LLM"))

if option == "Evaluate LLM":
    # Input fields for LLM evaluation
    st.subheader("Input Data")
    question = st.text_input("Question", "Question to be asked from LLM/RAG application")
    context = st.text_area("Reference Context (top 'k' documents)", "Enter Original answer")
    generated_output = st.text_area("LLM Generated Output", "LLM Answer")

    # Evaluation button
    if st.button("Evaluate"):
        if question and context and generated_output:
            st.subheader("Evaluation Results")

            # Perform evaluations
            metrics = evaluator.evaluate_all(generated_output, context)

            # Display metrics with explanations
            st.write(f"**BLEU Score**: {metrics['BLEU']}")
            st.write("BLEU measures the overlap between the generated output and reference text based on n-grams. Higher scores indicate better match.")

            st.write(f"**ROUGE-1 Score**: {metrics['ROUGE-1']}")
            st.write("ROUGE-1 measures the overlap of unigrams between the generated output and reference text. Higher scores indicate better match.")

            st.write(f"**BERT Precision**: {metrics['BERT P']}")
            st.write(f"**BERT Recall**: {metrics['BERT R']}")
            st.write(f"**BERT F1 Score**: {metrics['BERT F1']}")
            st.write("BERTScore evaluates the semantic similarity between the generated output and reference text using BERT embeddings.")

            st.write(f"**Perplexity**: {metrics['Perplexity']}")
            st.write("Perplexity measures how well a language model predicts the text. Lower values indicate better fluency and coherence.")

            st.write(f"**Diversity**: {metrics['Diversity']}")
            st.write("Diversity measures the uniqueness of bigrams in the generated output. Higher values indicate more diverse and varied output.")

            st.write(f"**Racial Bias**: {metrics['Racial Bias']}")
            st.write("Racial Bias score indicates the presence of biased language in the generated output. Higher scores indicate more bias.")

            st.write(f"**Burstiness**: {metrics['Burstiness']}")
            st.write("Burstiness measures the repetition frequency of the most frequent token in the text. Lower values indicate more uniform token distribution.")

            st.write(f"**Readability**: {metrics['Readability']}")
            st.write("Readability is measured by the Flesch-Kincaid readability score. Higher values indicate easier readability.")

        else:
            st.write("Please provide all inputs to evaluate.")
