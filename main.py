# import os
import csv
import logging
from os import getenv
import time
import json
import openai
import argparse
from environs import Env
import pandas as pd
import tiktoken
from sklearn.model_selection import StratifiedShuffleSplit
from rich import print


def argument_parser():
    parser = argparse.ArgumentParser(description="Parser to read a filename from the command.")
    parser.add_argument("-f", "--file",
                        help="Path and filename containing the xml-file. E.g. \\data\\xml_file.xml",
                        required=True)
    parser.add_argument("-v", "--verbose",
                        action="store_true",
                        help="Whether to display logging information.")
    parser.add_argument("-m", "--mode",
                        help="Mode of analysis. Possible options: 1st_rd, 2nd_rd",
                        choices=["1st_rd", "2nd_rd"],
                        required=True)

    return parser


def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def send_messages(messages, ids):
    results = []
    for message, single_id in zip(messages, ids):
        prompt_list = [
            "If the following comment related specifically to drones or uavs? Give 3-5 sentence explanation why it is or why it is not. Comment: ",
            "Is the following statement true of false: 'The following Explanation states that the Comment is related to drones or uavs.' Answer true if it does and false if it does not. Give answer only as a true/false boolean value. 'Explanation': ",
            "Read the following 'Explanation' and 'Comment' and give a confidence score on how correct the explanation is in relation to the comment. Return ONLY an integer on a scale from 1 to 5 where 1 means you are not confident at all of the connection of 'Explanation' and 'Comment' and 5 is that you are completely confident of the connection of 'Explanation' and 'Comment'."]
        for i, prompt in enumerate(prompt_list):
            retry_count = 0
            retry_limit = 3
            completed = False
            completion_list = []
            response_list = []
            completion_list.append({"role": "system", "content": "You are a helpful assistant."})
            if i == 0:
                completion_list.append({"role": "user", "content": prompt + message})
            elif i == 1:
                completion_list.append({"role": "user", "content": prompt + result["Explanation"]})
            else:
                completion_list.append({"role": "user", "content": prompt + " 'Explanation': " + result[
                    "Explanation"] + " 'Comment': " + result["Comment"]})
            response_list.append({"role": "assistant", "content": ""})

            while retry_count < retry_limit and not completed:
                try:
                    individual_response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=completion_list,
                        max_tokens=500
                    )
                    completed = True  # Set the flag to indicate successful completion
                except openai.error.RateLimitError:
                    logging.info(f"Rate limit exceeded. Retrying... (Attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(0.2)
                except openai.error.APIError:
                    logging.info(f"API Error. Retrying... (Attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(0.2)
                except openai.error.ServiceUnavailableError:
                    logging.info(f"The server is overloaded or not ready yet. Retrying... (Attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(0.2)
                except openai.error.APIConnectionError:
                    logging.info(f"Connection error. Retrying... (Attempt {retry_count + 1})")
                    retry_count += 1
                    time.sleep(0.2)
            if completed:
                time.sleep(0.2)
                for choice in individual_response['choices']:
                    if 'message' in choice and 'role' in choice['message']:
                        if choice['message']['role'] == 'assistant':
                            current_response = choice['message']['content'].strip()
                if i == 0:
                    result = {"Explanation": current_response, "Comment": message, "Combined ID": single_id}
                elif i == 1:
                    result["Related"] = current_response
                else:
                    result["Confidence"] = current_response
        results.append(result)

    return results


def send_messages_maldonado(prompt_message, messages, id_parameter, ismaldo=False):
    results = []
    i = 0
    for message, single_classification in zip(messages, id_parameter):
        retry_count = 0
        retry_limit = 50
        completed = False
        completion_list = []
        response_list = []
        completion_list.append({"role": "system", "content": "You are a helpful assistant."})
        completion_list.append({"role": "user", "content": prompt_message + message})
        response_list.append({"role": "assistant", "content": ""})
        message_length = 3000

        while retry_count < retry_limit and not completed:
            try:
                individual_response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=completion_list,
                    max_tokens=message_length
                )
                completed = True  # Set the flag to indicate successful completion
            except openai.error.RateLimitError:
                logging.info(f"Rate limit exceeded. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.2)
            except openai.error.APIError:
                logging.info(f"API Error. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.2)
            except openai.error.ServiceUnavailableError:
                logging.info(f"The server is overloaded or not ready yet. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.2)
            except openai.error.Timeout:
                logging.info(f"Request timed out. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(2)
            except openai.error.APIConnectionError:
                logging.info(f"API Connection Error. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.3)
            except openai.error.InvalidRequestError:
                logging.info(
                    f"The message length is too long, reducing the length. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.2)
                message_length = message_length - 100
        if completed:
            time.sleep(0.2)
            i += 1
            if i % 10 == 0:
                logging.info(f'Processed {i} comments.')
            for choice in individual_response['choices']:
                if 'message' in choice and 'role' in choice['message']:
                    if choice['message']['role'] == 'assistant':
                        current_response = choice['message']['content'].strip()
            if ismaldo:
                result = {"Message": message,
                          "Classification": current_response,
                          "Correct_Classification": single_classification}
            else:
                result = {"Comment": message,
                          "Classification": current_response,
                          "Comment_Id": single_classification}
        results.append(result)

    return results


def send_messages_with_guardrails(messages, guard, ids):
    assistant_responses = []
    missed_messages = []
    i = 0
    for message, single_id in zip(messages, ids):
        retry_count = 0
        retry_limit = 20
        completed = False

        while retry_count < retry_limit and not completed:
            try:
                raw_llm_output, validated_output = guard(
                    openai.ChatCompletion.create,
                    prompt_params={"document": message},
                    model="gpt-3.5-turbo",
                    max_tokens=1000,
                    temperature=0,
                )

            except openai.error.RateLimitError:
                logging.info(f"Rate limit exceeded. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.3)
            except openai.error.APIError:
                logging.info(f"API Error. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.3)
            except openai.error.Timeout:
                logging.info(f"Request timed out. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(2)
            except openai.error.APIConnectionError:
                logging.info(f"API Connection Error. Retrying... (Attempt {retry_count + 1})")
                retry_count += 1
                time.sleep(0.3)

            if validated_output is not None:
                completed = True  # Set the flag to indicate successful completion
            else:
                logging.info("Validated input was None. Saving to missed list.")
                retry_count += 1
                time.sleep(0.3)
                missed_messages.append((single_id, message))
        if completed:
            time.sleep(0.2)
            validated_output['message'] = message
            validated_output['combined_id'] = single_id
            assistant_responses.append(validated_output)
            i += 1
            logging.info(f'Processed {i} comments.')


        if i % 100 == 0:
            with open(f'./temp_save/5k_htf_td_classification_{i}.json', 'w', encoding='utf-8') as f:
                json.dump(assistant_responses, f, ensure_ascii=False, indent=4)

            sample_df = pd.DataFrame(assistant_responses)
            sample_df.to_csv(path_or_buf=f'./temp_save/5k_htf_td_classification_{i}.csv', sep=';', index=False)
            print(f'Saved the comments up to {i}.')
    if len(missed_messages) > 0:
        with open('missed_ids_and_messages.csv', 'w', encoding='utf-8', newline='') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr.writerow(missed_messages)
    logging.info(f"Total number of failed messages: {len(missed_messages)}.")

    return assistant_responses


def remove_single_occurrence_groups(df, strata_column):
    # Get counts for each group in the stratified column
    group_counts = df[strata_column].value_counts()

    # Identify groups with only 1 occurrence
    single_occurrence_groups = group_counts[group_counts == 1].index

    # Remove these groups from the DataFrame
    df = df[~df[strata_column].isin(single_occurrence_groups)]

    return df


def stratified_sample(df, strata_column, sample_size, random_state):
    # Calculate the fractions of each stratum
    stratum_proportions = df[strata_column].value_counts(normalize=True)

    # Initialize the StratifiedShuffleSplit instance
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_size, random_state=random_state)

    # Get indices for stratified sample
    for _, sample_indices in sss.split(df, df[strata_column]):
        stratified_sample = df.iloc[sample_indices]

    return stratified_sample


def clean_2nd_rd_dataframe(df):
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    # mask = ~((df["Human_Suggestion"].astype(str).str.contains(df["ChatGPT_R1"].astype(str))) | (df["ChatGPT_R1"].astype(str).str.contains(df["Human_Suggestion"].astype(str))))
    # Filter rows based on mask
    # df = df[mask]
    df = df.drop(df[(df.Human_Suggestion == "No Debt") & (df.ChatGPT_R1 == "No")].index)
    df = df.drop(df[(df.Human_Suggestion == df.ChatGPT_R1)].index)

    mask = df.apply(lambda row: not (str(row["Human_Suggestion"]) in str(row["ChatGPT_R1"]) or str(row["ChatGPT_R1"]) in str(row["Human_Suggestion"])), axis=1)

    # Return the filtered DataFrame
    return df[mask]

    # df = df[df['Human_Suggestion'].str.contains(df['ChatGPT_R1'], regex=False)]
    #return df


def create_prompt_list_from_df(df):
    prompt_series = "Comment: " + df['Comment'] + "\nCategories: " + df['Human_Suggestion'] + "/" + df['ChatGPT_R1']
    return prompt_series.tolist()


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    Env.read_env()
    openai.api_key = getenv("OPENAI_API_KEY")

    if args.mode == "1st_rd":
        ### Read the SATD-csv.
        messages_df = pd.read_csv(filepath_or_buffer=args.file, sep=";")
        # messages_df = messages_df.loc[messages_df['SATD'] == 1]
        messages_df = messages_df[['Project_Name', 'combined_id', 'Combined_Comment_Text']]
        messages_df['combined_id'] =messages_df['combined_id'].astype(int)
        messages_df = remove_single_occurrence_groups(df=messages_df, strata_column='Project_Name')
        logging.info("Dataset loaded.")

        # Creating a test sample of 385 cases to see, whether ChatGPT can be relied on
        # sample_df = stratified_sample(messages_df, 'Project_Name', 385, 26)
        # sample_df['combined_id'] = sample_df['combined_id'].astype(int)
        # logging.info("Sample created.")

        td_classification_prompt_shorter_all = '''Comment can be categorised as belonging to one of the following categories:
        Architectural Debt, Build Debt, Code Debt, Design Debt, Defect Debt, Documentation Debt, Requirement Debt, Test Debt, Unknown Debt or No Debt.
        
        A comment containing a keyword TODO, FIXME, HACK or XXX are always classified into one of the categories besides No Debt. 
    
        Architectural Debt Explanation: Architectural Debt refers to problems encountered in project architecture, for example, violation of modularity, which can
        affect architectural requirements (performance, robustness, among others), or architecture decisions that make
        compromises in some internal quality aspects, such as maintainability.
    
        Build Debt Explanation: Build Debt refers to build related issues that make this task harder, and more 
        time/processing consuming unnecessarily, including unnecessary code and ill-defined dependencies making the process 
        slower than it should be, or to flaws in a software system, in its build system,
        or in its build process that make the build overly complex and difficult.
    
        Code Debt Explanation: Code Debt refers to problems found in the source code which can affect negatively the legibility
        of the code making it more difficult to be maintained, including issues related to bad coding practices, 
        or to poorly written code that violates best coding practices or coding rules. Examples include code duplication and overly complex code.
    
        Design Debt Explanation: Design Debt refers to technical shortcuts that are taken in detailed design or use of practices
        which violate the principles of good (object-oriented) design.
    
        Defect Debt Explanation: Defect Debt refers to defects, bugs, or failures found in software systems or known defects
        but due to competing priorities, and limited resources have to be deferred to a later time.
    
        Documentation Debt Explanation: Documentation Debt is insufficient, incomplete, missing, inadequate incomplete or outdated documentation.
    
        Requirement Debt Explanation: Requirement Debt is the distance between the optimal requirements specification and the actual system implementation 
        or tradeoffs made with respect to what requirements the development team need to implement or how to implement them.
    
        Test Debt Explanation: Test Debt is shortcuts taken in testing or issues found in testing activities which can affect the quality of testing activities.
    
        Unknown Debt Explanation: Unknown Debt refers to issues in the code, which does not fall into any of the other categories. One example of Unknown Debt is a lone keyword of TODO, FIXME, HACK or XXX without any explanation.
        
        No Debt Explanation: No Debt refers to cases, where none of the other categories match, and the comment does not refer to any issues.
    
        Following these examples, categorise the following comment as either Architectural Debt, Build Debt, Code Debt, Design Debt, Defect Debt, Documentation Debt, Requirement Debt, Test Debt, Unknown Debt or No Debt. 
    
        Return the answer in following format: 
    
        Explanation: Short, 3-5 sentence explanation on the classification.
    
        Classification: Architectural Debt, Build Debt, Code Debt, Design Debt, Defect Debt, Documentation Debt, Requirement Debt, Test Debt, Unknown Debt or No Debt.
    
        The Comment to be classified is:
        '''

        responses = send_messages_maldonado(prompt_message=td_classification_prompt_shorter_all,
                                            messages=messages_df['Combined_Comment_Text'].tolist(),
                                            id_parameter=messages_df['combined_id'].tolist())

        with open('3k_td_classificationv2.json', 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        sample_df = pd.DataFrame(responses)
        sample_df.to_csv(path_or_buf='3k_td_classificationv2.csv', sep=';', index=False)
    elif args.mode == "2nd_rd":
        ### Read the annotated SATD-csv after the middle check in R.
        messages_df = pd.read_csv(filepath_or_buffer=args.file, sep=";")
        messages_df["ChatGPT_R2"] = "Not_Done"
        logging.info("Dataset loaded.")
        messages_df_cleaned = clean_2nd_rd_dataframe(df=messages_df)

        # Merging the results together and rearranging to original row order
        messages_df = messages_df.drop(messages_df.index[messages_df_cleaned.index])
        messages_merged = pd.concat([messages_df, messages_df_cleaned], ignore_index=False)
        messages_merged = messages_merged.sort_index(ascending=True)
        # Create the end of the prompt
        prompt_list = create_prompt_list_from_df(df=messages_df_cleaned)
        logging.info("Prompt list created.")


        td_2nd_rd_classification_prompt = '''Comment can be categorised as belonging to one of the following categories:
                Architectural Debt, Build Debt, Code Debt, Design Debt, Defect Debt, Documentation Debt, Requirement Debt, Test Debt, Unknown Debt or No Debt.

                A comment containing a keyword TODO, FIXME, HACK or XXX are always classified into one of the categories besides No Debt. 

                Architectural Debt Explanation: Architectural Debt refers to problems encountered in project architecture, for example, violation of modularity, which can
                affect architectural requirements (performance, robustness, among others), or architecture decisions that make
                compromises in some internal quality aspects, such as maintainability.

                Build Debt Explanation: Build Debt refers to build related issues that make this task harder, and more 
                time/processing consuming unnecessarily, including unnecessary code and ill-defined dependencies making the process 
                slower than it should be, or to flaws in a software system, in its build system,
                or in its build process that make the build overly complex and difficult.

                Code Debt Explanation: Code Debt refers to problems found in the source code which can affect negatively the legibility
                of the code making it more difficult to be maintained, including issues related to bad coding practices, 
                or to poorly written code that violates best coding practices or coding rules. Examples include code duplication and overly complex code.

                Design Debt Explanation: Design Debt refers to technical shortcuts that are taken in detailed design or use of practices
                which violate the principles of good (object-oriented) design.

                Defect Debt Explanation: Defect Debt refers to defects, bugs, or failures found in software systems or known defects
                but due to competing priorities, and limited resources have to be deferred to a later time.

                Documentation Debt Explanation: Documentation Debt is insufficient, incomplete, missing, inadequate incomplete or outdated documentation.

                Requirement Debt Explanation: Requirement Debt is the distance between the optimal requirements specification and the actual system implementation 
                or tradeoffs made with respect to what requirements the development team need to implement or how to implement them.

                Test Debt Explanation: Test Debt is shortcuts taken in testing or issues found in testing activities which can affect the quality of testing activities.

                Unknown Debt Explanation: Unknown Debt refers to issues in the code, which does not fall into any of the other categories. One example of Unknown Debt is a lone keyword of TODO, FIXME, HACK or XXX without any explanation.

                No Debt Explanation: No Debt refers to cases, where none of the other categories match, and the comment does not refer to any issues.

                I will give you a Comment and the only possible categories it could belong to. Consider and explain each category separately why it would fit. Finally, choose only one category as your answer, the one which fits the best.

                Return the answer in following format: 

                Explanation: Short, 3-5 sentence explanation on the classification.
                Classification: Architectural Debt, Build Debt, Code Debt, Design Debt, Defect Debt, Documentation Debt, Requirement Debt, Test Debt, Unknown Debt or No Debt. Category must match one of the provided ones.
                
                '''

        responses = send_messages_maldonado(prompt_message=td_2nd_rd_classification_prompt,
                                            messages=prompt_list,
                                            id_parameter=messages_df_cleaned['Comment_Id'].tolist())

        with open('3k_td_classification_AI1_H1_AI2_done.json', 'w', encoding='utf-8') as f:
            json.dump(responses, f, ensure_ascii=False, indent=4)

        sample_df = pd.DataFrame(responses)
        sample_df.to_csv(path_or_buf='3k_td_classification_AI1_H1_AI2_done.csv', sep=';', index=False)
