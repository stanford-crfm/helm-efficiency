import json
import numpy as np
from sklearn.linear_model import LinearRegression


def solve_linear_regression(num_output_tokens_and_runtimes,
                            print_r_squared=False):
    X = [[x] for (x, _) in num_output_tokens_and_runtimes]
    Y = [y for (_, y) in num_output_tokens_and_runtimes]
    reg = LinearRegression()
    reg.fit(X, Y)
    if print_r_squared:
        print(f"R^2 score: {reg.score(X, Y):.3f}")
    return round(reg.coef_[0], 3), round(reg.intercept_, 3)


def compute_best_fit(runtimes, models,
                     name_mapping,
                     estimate_runtime_for_prompt_tokens=False,
                     real_system=False, filename=None):
    json_obj = {}
    for model in models:
        model_obj = {}
        model.replace("-", "_")
        
        # Get all unique num_prompt_tokens.
        all_num_prompt_tokens = set()
        for label in runtimes:
            if label[0] == model:
                all_num_prompt_tokens.add(label[1])
        all_num_prompt_tokens = sorted(list(all_num_prompt_tokens))
        
        # Group runtimes by num_prompt_tokens.
        processed_runtimes = [[] for num_prompt_tokens in
                              all_num_prompt_tokens]
        for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
            for label, runtime in runtimes.items():
                if label[0] == model and label[1] == num_prompt_tokens:
                    processed_runtimes[i].append((label[2], runtime))
            processed_runtimes[i].sort()
           
        runtime_for_prompt_tokens = {}
        if estimate_runtime_for_prompt_tokens:
            for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
                num_output_tokens_and_runtimes = []
                for label, runtime in runtimes.items():
                    if label[0] == model and label[1] == num_prompt_tokens:
                        num_output_tokens_and_runtimes.append(
                            (label[2], runtime))
                slope, y_intercept = solve_linear_regression(
                    num_output_tokens_and_runtimes)
                runtime_for_prompt_tokens[num_prompt_tokens] = round(
                    slope + y_intercept, 3)
                if i > 0:
                    runtime_for_prompt_tokens[num_prompt_tokens] = max(
                        runtime_for_prompt_tokens[num_prompt_tokens],
                        runtime_for_prompt_tokens[all_num_prompt_tokens[0]]
                    )
        else:
            # Cost of embedding num_prompt_tokens is just the runtime when
            # num_output_tokens is equal to 1.
            for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
                assert processed_runtimes[i][0][0] == 1
                runtime_for_prompt_tokens[num_prompt_tokens] = round(
                    processed_runtimes[i][0][1], 3)

        # Perform regression between adjusted runtimes (end-to-end
        # runtime minus embedding runtime with num_prompt_tokens) and
        # number of output tokens, to obtain the runtime per generated
        # output token.
        num_output_tokens_and_runtimes = []
        for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
            for label, runtime in runtimes.items():
                if label[0] == model and label[1] == num_prompt_tokens:
                    num_output_tokens_and_runtimes.append(
                        (label[2], runtime - runtime_for_prompt_tokens[
                            num_prompt_tokens]))
        runtime_per_output_token, _ = solve_linear_regression(
            num_output_tokens_and_runtimes, print_r_squared=True)
        overhead = round(runtime_for_prompt_tokens[all_num_prompt_tokens[0]]
                         - runtime_per_output_token, 3)
        if real_system:
            for num_prompt_tokens in runtime_for_prompt_tokens:
                runtime_for_prompt_tokens[num_prompt_tokens] -= overhead
                runtime_for_prompt_tokens[num_prompt_tokens] = round(
                    runtime_for_prompt_tokens[num_prompt_tokens], 3)

        print(f"{model}:\n\tRuntime per output token = "
              f"{runtime_per_output_token:.3f} seconds")
        print(f"\tOverhead: {overhead:.3f} seconds")

        for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
            runtime = runtime_for_prompt_tokens[num_prompt_tokens]
            print(f"\tRuntime for {num_prompt_tokens} prompt "
                  f"token(s) = {runtime:.3f} seconds")
        print()
        
        square_errors = []
        for i, num_prompt_tokens in enumerate(all_num_prompt_tokens):
            for label, runtime in runtimes.items():
                if label[0] == model and label[1] == num_prompt_tokens:
                    estimated_runtime = runtime_for_prompt_tokens[
                        num_prompt_tokens] + (
                        (label[2] - 1) * runtime_per_output_token)
                    square_errors.append((runtime - estimated_runtime)**2)
        print(f"\tMean squared error: {np.mean(square_errors):.4f}")
        print(f"\tMaximum squared error: {max(square_errors):.4f}")
        print()

        model_obj['runtime_per_output_token'] = runtime_per_output_token
        model_obj['runtime_for_prompt_tokens'] = runtime_for_prompt_tokens
        if real_system:
            model_obj['overhead'] = overhead
                
        if model in name_mapping:
            model_names = name_mapping[model]
            if real_system:
                model_names = model_names[:1]
            for model_name in model_names:
                json_obj[model_name] = model_obj
    print("=" * 100)
    print()
    
    if filename is not None:
        with open(filename, 'w') as f:
            json.dump(json_obj, f, indent=2)
            
    return json_obj