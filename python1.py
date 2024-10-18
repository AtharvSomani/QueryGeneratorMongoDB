import re
import pandas as pd
from pymongo import MongoClient
from transformers import T5Tokenizer, T5ForConditionalGeneration

client = MongoClient('mongodb://localhost:27017/')
db = client['product_database']  # Name of your MongoDB database
collection = db['products']  # Name of your MongoDB collection


model_name = "t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def parse_generated_query(input_description, generated_query):
    query = {}

    # Regex patterns for common conditions (like greater than, less than, equal to)
    patterns = {
        'gt': re.compile(r"(\w+)\s*greater than\s*(\d+)"),
        'lt': re.compile(r"(\w+)\s*less than\s*(\d+)"),
        'eq': re.compile(r"(\w+)\s*equal to\s*['\"]?(\w+)['\"]?")
    }

    # Apply regex patterns to extract fields and values from the input description (fallback mechanism)
    for condition, pattern in patterns.items():
        matches = pattern.findall(input_description)
        for match in matches:
            field, value = match
            field = field.strip()

            if condition == 'gt':
                query[field] = {"$gt": float(value)}
            elif condition == 'lt':
                query[field] = {"$lt": float(value)}
            elif condition == 'eq':
                query[field] = value

    # Add more validation and post-process the generated_query to handle errors.
    # Check if the T5 output is reasonable
    if 'ReviewCount' not in generated_query:
        return query
    else:
        return query  # Fallback to regex-based parsing


# Function to generate query using T5 model
def generate_query_with_t5(input_description):
    # Add schema to the input prompt
    schema = """
    Schema: ProductID, ProductName, Category, Price, Rating, ReviewCount, Stock, Discount, Brand, LaunchDate
    """
    # Include schema in the input description
    prompt = f"Create a MongoDB query for the following schema: {schema} Query: {input_description}"
    print(len(prompt))

    # Encode the input description
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    print(inputs)
    print(f"Input IDs size: {inputs['input_ids'].size()}")



    outputs = model.generate(inputs.input_ids, max_length=150, num_beams=5, early_stopping=True)
    print(outputs)
    print (outputs.size())
    # Decode the generated query
    generated_query = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"Generated Query (T5 output): {generated_query}")

    # Parse the generated query to a valid MongoDB format
    parsed_query = parse_generated_query(input_description, generated_query)
    return parsed_query


def fetch_and_present_data(query, save_csv=False, csv_filename='output.csv', sort_by=None, sort_order=None):
    if query is None or len(query) == 0:
        print("Invalid query generated. Please try again.")
        return

    if sort_by:
        sort_order = -1 if sort_order == "desc" else 1
        results = collection.find(query).sort(sort_by, sort_order)
    else:
        results = collection.find(query)

    df = pd.DataFrame(list(results))

    if df.empty:
        print("No data found for the given query.")
        return


    if '_id' in df.columns:
        df = df.drop(columns=['_id'])

    # Remove duplicates
    df = df.drop_duplicates()

    if save_csv:
        df.to_csv(csv_filename, index=False)
        print(f'Data saved to {csv_filename} without duplicates.')
    else:
        print(df)


def main():
    # Ask user for input description
    input_description = input(
        "Enter a query description (e.g., 'products with price greater than 50 and rating equal to 4.5'): ")

    # Generate MongoDB query using T5 model
    query = generate_query_with_t5(input_description)

    # Print the generated query for debugging
    print(f"Parsed Query: {query}")

    # Fetch and present data
    save_option = input("Do you want to save the result to a CSV file? (yes/no): ").lower()
    save_csv = save_option == 'yes'

    if save_csv:
        csv_filename = input("Enter the CSV filename (e.g., 'result.csv'): ")
        fetch_and_present_data(query, save_csv=True, csv_filename=csv_filename)
    else:
        fetch_and_present_data(query, save_csv=False)


if __name__ == "__main__":
    main()
