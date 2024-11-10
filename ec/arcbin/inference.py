# inference.py

import os
import sys
import json
import dill
import numpy as np
import torch
import time
import binutil
import datetime
from dreamcoder.domains.arc.makeTasks import convert_arc_task
from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from arckit import Task
from arckit.data import TaskSet
from dreamcoder.utilities import numberOfCPUs
from dreamcoder.domains.arc import arcPrimitivesIC2

# Assuming that arcPrimitivesIC2.dsl contains your primitives and generate_ocaml_primitives method
primitives = arcPrimitivesIC2.dsl.primitives.values()
arcPrimitivesIC2.dsl.generate_ocaml_primitives()

# Path to your trained result pickle file
RESULT_PATH = '../results/result_bai0os3w_0.pkl'  # Update with your actual path
# Path to your sample submission file
SAMPLE_SUBMISSION_PATH = '../data/sample_submission.json'  # Update with your actual path
TEST_PATH = '../data/arc-agi_test_challenges.json'  # Update with your actual path
# SAMPLE_SUBMISSION_PATH = '/kaggle/input/arc-prize-2024/sample_submission.json'  # Update with your actual path
SUBMISSION_PATH = '../data/submission.json'
    

# inference.py
# Load your pre-trained grammar and recognition model
def load_trained_model(result_path):
    with open(result_path, 'rb') as f:
        result = dill.load(f)
    grammar = result.grammars[-1]  # Use the latest grammar
    recognition_model = result.recognitionModel  # Load the recognition model
    return grammar, recognition_model

# Load test tasks from the JSON file and convert them into Task objects
# def load_test_tasks(test_path):
#     with open(test_path, 'r') as f:
#         test_data = json.load(f)
    
#     test_tasks = []
#     for task_id, task_dict in test_data.items():
#         # Extract examples from the 'train' section (if any)
#         examples = []
#         for ex in task_dict.get('train', []):
#             input_grid = ex['input']
#             output_grid = ex['output']
#             examples.append(((input_grid,), output_grid))
        
#         # Extract test inputs from the 'test' section
#         test_examples = []
#         for ex in task_dict.get('test', []):
#             input_grid = ex['input']
#             # Since we don't have outputs for test inputs, set output to None
#             output_grid = None
#             test_examples.append(((input_grid,), output_grid))

#         # Create a Task object
#         task = Task(
#             name=task_id,
#             examples=examples,
#             test_examples=test_examples,
#             request=test_examples
#         )
        
#         test_tasks.append(task)
    
#     return test_tasks
def load_data_kaggle() -> (TaskSet):
    data = json.load(open("../data/arc-agi_test_challenges.json"))
    train_tasks = []
    eval_tasks = []
    for id, task in data.items():
        
        for i in range(len(task['test'])):
            task['test'][i]['output'] = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
        
        for train,test in zip(task['train'],task['test']):
            if not train.keys() == {'input', 'output'}:
                assert False, f"train {id} task {train} has keys {train.keys()}"
            if not test.keys() == {'input', 'output'}:
                assert False, f"test {id} task {test} has keys {test.keys()}"
                
        train_tasks.append(Task(id, task['train'], task['test'], 'test'))

    return TaskSet(train_tasks)

def get_arc_tasks(n=None):
    dataset = load_data_kaggle()
    if n:
        dataset = dataset[:n]
    return [convert_arc_task(task) for task in dataset]

# Perform inference on the test tasks
def infer_test_tasks(grammar, recognition_model, test_tasks,testingTasks, output_prefix='./experimentOutputs/arc/', CPUs=numberOfCPUs()):
    # Create a directory to store any outputs if necessary
    os.makedirs(output_prefix, exist_ok=True)
    # Set up arguments for ecIterator
    args = {
        'grammar': grammar,
        'tasks': test_tasks,
        'testingTasks': testingTasks,
        'iterations': 1,
        'useDSL': True,
        'useRecognitionModel': True,
        'featureExtractor': recognition_model.featureExtractor,
        'recognizer': recognition_model,
        'enumerationTimeout': 60,  # Adjust as necessary
        'CPUs': CPUs,
        'outputPrefix': output_prefix,
        'evaluationTimeout': 60,
        'testingTimeout': 0,
        'maximumFrontier': 5,
        'solver': 'python',
        'compressor': 'ocaml',
        'taskBatchSize' : 100,
        'recognitionTimeout': 60,
        'cuda':True
    }

    # Run ecIterator for inference
    generator = ecIterator(**args)

    # Collect the results
    for result in generator:
        # result.taskSolutions will contain the solutions for each task
        return result.taskSolutions

# Generate the submission file
def generate_submission(task_solutions, sample_submission_path, submission_path):
    # Load the sample submission file
    with open(sample_submission_path, 'r') as f:
        submissions = json.load(f)

    for task in task_solutions:
        task_name = task.name
        frontier = task_solutions[task]
        if len(frontier.entries) > 0:
            # Get the best program
            best_entry = frontier.entries[0]
            program = best_entry.program

            # Execute the program on the test input(s)
            test_inputs = [ex[0][0] for ex in task.test_examples]  # List of input grids
            predictions = []
            for idx, input_grid in enumerate(test_inputs):
                try:
                    # Evaluate the program
                    output_grid = program.evaluate([], input_grid)
                    # Convert the output grid to a list of lists
                    if isinstance(output_grid, np.ndarray):
                        output_grid_list = output_grid.tolist()
                    else:
                        output_grid_list = output_grid
                    predictions.append(output_grid_list)
                except Exception as e:
                    print(f"Error executing program for task {task_name} on input {idx}: {e}")
                    # Handle execution failure
                    input_grid_list = input_grid.tolist() if isinstance(input_grid, np.ndarray) else input_grid
                    predictions.append(input_grid_list)  # Using input as output as a placeholder

            # Update the submission file with predictions
            if task_name in submissions:
                # Update each attempt in the submission
                for i in range(len(submissions[task_name])):
                    submissions[task_name][i]["attempt_1"] = predictions[i].grid.tolist()
                    submissions[task_name][i]["attempt_2"] = predictions[i].grid.tolist()
            else:
                print(f"Task {task_name} not found in sample submission.")
        else:
            print(f"No solution found for task {task_name}.")
            # Handle missing solution
            test_inputs = [ex[0][0] for ex in task.test_examples]
            predictions = []
            for input_grid in test_inputs:
                input_grid_list = input_grid.tolist() if isinstance(input_grid, np.ndarray) else input_grid
                predictions.append(input_grid_list)  # Using input as output as a placeholder

            # Update the submission file with placeholder outputs
            if task_name in submissions:
                for i in range(len(submissions[task_name])):
                    submissions[task_name][i]["attempt_1"] = predictions[i].grid.tolist()
                    submissions[task_name][i]["attempt_2"] = predictions[i].grid.tolist()
            else:
                print(f"Task {task_name} not found in sample submission.")
    # Write the updated submission file
    with open(submission_path, 'w') as f:
        json.dump(submissions, f, indent=4)
    print(f"Submission file saved to {submission_path}")

def main():
    # Load the trained model
    print("Loading trained model...")
    grammar, recognition_model = load_trained_model(RESULT_PATH)

    # Load test tasks
    print("Loading test tasks...")
    with open(TEST_PATH, 'r') as f:
        test_data = json.load(f)
    testingTasks = list(test_data.keys())
    test_tasks = get_arc_tasks()
    print(f"Loaded {len(test_tasks)} test tasks.")

    # Perform inference
    print("Performing inference on test tasks...")
    # import IPython; IPython.embed();exit()
    task_solutions = infer_test_tasks(grammar, recognition_model, test_tasks,testingTasks=testingTasks)

    # Generate submission file
    print("Generating submission file...")
    generate_submission(task_solutions, SAMPLE_SUBMISSION_PATH, SUBMISSION_PATH)

if __name__ == '__main__':
    print(datetime.datetime.now())
    main()
    print(datetime.datetime.now())
    