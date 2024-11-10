import dill
RESULT_PATH = '../results/result_bai0os3w_0.pkl'  # Update with your actual path

def load_trained_model(result_path):
    with open(result_path, 'rb') as f:
        result = dill.load(f)
    grammar = result.grammars[-1]  # Use the latest grammar
    recognition_model = result.recognitionModel  # Load the recognition model
    return grammar, recognition_model

load_trained_model(RESULT_PATH)