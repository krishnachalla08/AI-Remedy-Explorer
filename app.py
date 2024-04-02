from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import PyPDF2
from nltk.tokenize import sent_tokenize
import spacy

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")


def load_pdf_text(path):
    """Extracts text from a PDF."""
    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(reader.pages):
            text += f"PAGE_START_{page_num}\n"  # Add page delimiter
            text += page.extract_text()
            text += f"\nPAGE_END_{page_num}\n"  # Add page delimiter
    return text


def build_inverted_index(text):
    """Creates an inverted index from the text."""
    index = {}
    current_page = None
    for line in text.splitlines():
        if line.startswith('PAGE_START_'):
            current_page = int(line.split('_')[-1])
        elif line.startswith('PAGE_END_'):
            current_page = None
        else:
            for word in line.split():
                cleaned_word = ''.join(c for c in word if c.isalnum())
                if cleaned_word:
                    if cleaned_word.lower() in index:
                        index[cleaned_word.lower()].append((current_page, line.find(word)))
                    else:
                        index[cleaned_word.lower()] = [(current_page, line.find(word))]
    return index



def get_page_text(page_num, reader):
    """Extracts text from a specific page in the PDF."""
    return reader.pages[page_num].extract_text()


def answer_query(index, query, reader):
    """Provides a refined answer with improved information presentation and mention filtering."""
    keywords = [word.lower() for word in query.split()]

    relevant_pages = set()
    for keyword in keywords:
        if keyword in index:
            relevant_pages.update(page for page, _ in index[keyword])

    if not relevant_pages:
        return "No relevant information found in the PDF for your query."

    relevant_sentences = []
    for page_num in relevant_pages:
        page_text = get_page_text(page_num, reader)
        doc = nlp(page_text)

        for sentence in doc.sents:
            # Check for presence of all keywords in the sentence
            if all(keyword.lower() in sentence.text.lower() for keyword in keywords):
                relevant_sentences.append(sentence)

    if not relevant_sentences:
        return "No relevant sentences found in the PDF for your query."

    # Prepare the answer
    answer = f"Here are some relevant sentences from the PDF:\n"
    for sentence in relevant_sentences:
        answer += f"- {sentence.text}\n"

    return answer



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        pdf_path = request.form['pdf_path']
        query = request.form['query']

        pdf_text = load_pdf_text(pdf_path)
        index = build_inverted_index(pdf_text)

        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)

            answer = answer_query(index, query, reader)
            return render_template('index.html', query=query, answer=answer)

    return render_template('index.html', query='', answer='')


if __name__ == '__main__':
    app.run(debug=True)
