import binutil
import dill
import numpy as np
import os
import sys
import time
import json

from dreamcoder.dreamcoder import commandlineArguments, ecIterator
from dreamcoder.grammar import Grammar
from dreamcoder.domains.arc.makeTasks import get_arc_task, get_arc_tasks
from dreamcoder.domains.arc.main import MikelArcNet
from dreamcoder.domains.arc import arcPrimitivesIC2

import wandb

# Function to execute programs
def execute_program(program, input_grid):
    try:
        output_grid = program.evaluate(input_grid)
        return output_grid.grid
    except Exception as e:
        print(f'Error executing program: {e}')
        return None

# Function to evaluate solutions on tasks
def test_evaluate(task, soln):
    corrects_list = []
    n_test = len(task.test_examples)
    for i, frontier in enumerate(soln.entries):
        try:
            f = frontier.program.evaluate([])
        except Exception as e:
            print(f'Error evaluating program for task {task.name}: {e}')
            continue

        corrects = 0
        for (input_grid, ), output_grid in task.test_examples:
            try:
                program_output = f(input_grid)
                if hasattr(program_output, 'grid'):
                    corrects += np.array_equal(output_grid.grid, program_output.grid)
                else:
                    print(f'Program output for task {task.name} does not have attribute "grid".')
            except Exception as e:
                print(f'Exception {e} for task {task.name} with program: {frontier.program.body}')
        
        corrects_list.append(corrects)
    
    if n_test in corrects_list:
        correct_index = corrects_list.index(n_test)
        print(f'HIT @ {correct_index+1} for {task.name} with program: {soln.entries[correct_index].program.body}')
        return correct_index == 0, correct_index < 3
    else:
        print(f'FAIL: Evaluated {len(soln.entries)} solns for task {task.name}, no successes.')
        return False, False

if __name__ == "__main__":
    # Initialize primitives and grammar
    primitives = arcPrimitivesIC2.dsl.primitives.values()

    # Generate OCaml code for new primitives
    arcPrimitivesIC2.dsl.generate_ocaml_primitives()

    # Make a starting grammar to enumerate over (contains every primitive with uniform prior)
    grammar = Grammar.uniform(primitives)

    def extra_args(parser):
        parser.add_argument('--evalset', action='store_true', default=False, help='Use the eval set instead of the train set')
        parser.add_argument('--bothset', action='store_true', default=False, help='Use both datasets (800 tasks)')
        parser.add_argument('--task-isolation', action='store_true', default=False, help='Isolate tasks from each other')
        parser.add_argument('--inference-only', action='store_true', default=False, help='Run inference only without training')
        parser.add_argument('--grammar-path', type=str, default='grammar.pkl', help='Path to the saved grammar file for inference')

    # Generic command line options
    args = commandlineArguments(
        enumerationTimeout=120, 
        aic=0.1,
        iterations=1, 
        recognitionTimeout=360,
        featureExtractor=MikelArcNet,
        useRecognitionModel=True,  # True,
        # contextual=True,
        a=3, 
        maximumFrontier=10, 
        topK=5, 
        pseudoCounts=30.0,
        structurePenalty=0.1,
        solver='python',
        compressor='ocaml',
        CPUs=48,
        extras=extra_args,
    )

    # Initialize WandB
    wandb_config = args.copy()
    wandb_config['hostname'] = os.uname()[1]
    run = wandb.init(
        project="arc",
        config=wandb_config,
        save_code=True,
    )

    run_id = run.id  # int(time.time())
    print(f'Run ID: {run_id}')

    # Define metrics based on dataset selection
    run.define_metric('iteration')

    if args['evalset']:
        print('Running on eval-set')
        training = get_arc_tasks(n=400, eval=True)
        run.define_metric('test-hit1-eval', summary='max', goal='maximize', step_metric='iteration')
        run.define_metric('test-hit3-eval', summary='max', goal='maximize', step_metric='iteration')
    elif args['bothset']:
        print('Running on both sets')
        training = get_arc_tasks(n=400, eval=False) + get_arc_tasks(n=400, eval=True)
        run.define_metric('test-hit1-both', summary='max', goal='maximize', step_metric='iteration')
        run.define_metric('test-hit3-both', summary='max', goal='maximize', step_metric='iteration')
    else:
        print('Running on train-set')
        training = get_arc_tasks(n=400, eval=False)
        run.define_metric('test-hit1', summary='max', goal='maximize', step_metric='iteration')
        run.define_metric('test-hit3', summary='max', goal='maximize', step_metric='iteration')

    # Define additional metrics
    run.define_metric('batch')
    run.define_metric('recog-loss', summary='min', goal='minimise', step_metric='batch')
    run.define_metric('recog-mdl', summary='min', goal='minimise', step_metric='batch')
    run.define_metric('recog-class-loss', summary='min', goal='minimise', step_metric='batch')

    # Create output directory
    os.makedirs('./experimentOutputs/arc/', exist_ok=True)
    # print(training)
    # sys.exit()

    if not args['inference_only']:
        # Initialize the generator for training
        generator = ecIterator(grammar,
                               training,
                               testingTasks=[],
                               outputPrefix='./experimentOutputs/arc/',
                               **args)

        # Run the DreamCoder learning process for the set number of iterations
        for i, result in enumerate(generator):
            print('Test set evaluation')
            hit1, hit3 = 0, 0
            for task, soln in result.taskSolutions.items():
                if len(soln.entries) == 0:
                    continue
                try:
                    h1, h3 = test_evaluate(task, soln)
                except Exception as e:
                    print(f'Exception {e} while evaluating {task.name}')
                    h1, h3 = False, False
                hit1 += h1
                hit3 += h3

            print(f'Test summary: {hit1} ({(hit1/len(result.taskSolutions)*100):.1f}%) acc@1, {hit3} ({(hit3/len(result.taskSolutions)*100):.1f}%) acc@3')

            os.makedirs('results/', exist_ok=True)
            dill.dump(result, open(f'results/result_{run_id}_{i}.pkl', 'wb'))
            print('ecIterator count {}'.format(i))

            # Log metrics
            if args['evalset']:
                wandb.log({'test-hit1-eval': hit1, 'test-hit3-eval': hit3, 'iteration': i})
            elif args['bothset']:
                wandb.log({'test-hit1-both': hit1, 'test-hit3-both': hit3, 'iteration': i})
            else:
                wandb.log({'test-hit1': hit1, 'test-hit3': hit3, 'iteration': i})

    else:
        # Inference-only mode
        print('Running in inference-only mode.')

        # Adjust CPU allocation for inference to avoid multiprocessing issues
        args['CPUs'] = 1

        # Load pre-trained grammar
        grammar_path = args['grammar_path']
        if os.path.exists(grammar_path):
            print(f'Loading pre-trained grammar from {grammar_path}...')
            with open(grammar_path, "rb") as f:
                grammar = dill.load(f)
            print('Grammar loaded successfully.')
        else:
            print(f'Grammar file not found at {grammar_path}. Please ensure a pre-trained grammar is available.')
            sys.exit(1)

        # Define paths to your test and submission data
        test_path = '/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json'
        sample_path = '/kaggle/input/arc-prize-2024/sample_submission.json'

        # Load test tasks
        with open(test_path, "r") as f:
            test_tasks = json.load(f)

        # Prepare test tasks
        test_task_list = []
        for task_id, task_data in test_tasks.items():
            test_task = get_arc_task(task_id, task_data=task_data)  # Adjust if necessary
            test_task_list.append(test_task)

        # Create an iterator for test tasks
        test_generator = ecIterator(grammar,
                                    test_task_list,
                                    testingTasks=[],
                                    outputPrefix='./experimentOutputs/arc/test/',
                                    **args)

        # Dictionary to store predictions
        yq_predicts = {}

        for i, test_result in enumerate(test_generator):
            print(f'Processing test iteration {i}')

            for task, soln in test_result.taskSolutions.items():
                if len(soln.entries) == 0:
                    print(f'No solutions found for task {task.name}')
                    yq_predicts[task.name] = []  # Assign empty list if no solution
                    continue
                # Use the top solution
                top_soln = soln.entries[0].program
                # Execute the program on the test input
                try:
                    input_grid = task.test_examples[0][0].grid  # Assuming one test example per task
                except IndexError:
                    print(f'No test examples found for task {task.name}')
                    yq_predicts[task.name] = []
                    continue

                output_grid = execute_program(top_soln, input_grid)
                if output_grid is not None:
                    yq_predicts[task.name] = output_grid.tolist()
                else:
                    yq_predicts[task.name] = []  # or some default value

        # Align predictions to submission format
        with open(sample_path, "r") as f:
            submissions = json.load(f)

        for task_id, pred in yq_predicts.items():
            if task_id in submissions:
                for submission_entry in submissions[task_id]:
                    submission_entry["attempt_1"] = pred
                    submission_entry["attempt_2"] = pred  # Adjust if multiple attempts are needed
            else:
                print(f'Task ID {task_id} not found in submission format.')

        # Save submission file
        submission_path = "submission.json"
        with open(submission_path, "w") as f:
            json.dump(submissions, f, indent=4)

        print('Submission file saved to', submission_path)