from groq import Groq, GroqError
import json

# Set the API key directly
api_key = "*******"  # Replace with your actual API key

# Pre-defined relationships
pre_defined_rels = ["left", "right", "east", "south", "west", "north", "above", "below",
                    "behind", "front", "inside", "big", "small", "close", "furth", "between"]

def sentence_decompose(sentence):
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are a helpful assistant to help me decompose the sentence.
                                
                                Please don't change the expression in original sentence.
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                               Decompose the input sentence in the JSON format: subject, object, relationship.
                               
                               For the subject and object, remove the article e.g. a/are/the
                               
                               Only output the JSON format, don't output any explanation.
                               
                               The input sentence: '''{sentence}'''
                               """,
                }
            ],
            model="llama3-70b-8192",
            temperature=0  # Set temperature to 0
        )
        response = chat_completion.choices[0].message.content

        try:
            parsed_json = json.loads(response)
        except json.JSONDecodeError:
            raise ValueError("The response is not a valid JSON format")

        subject = parsed_json["subject"]
        object_ = parsed_json["object"]
        relationship = parsed_json["relationship"]

        return subject, object_, relationship

    except GroqError as e:
        print(f"An error occurred: {e}")
        return None


def relationship_analysis(relationship):
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are a helpful assistant to help me distinguish the spatial and semantic relationship
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                Tell me if the input phrase is spatial relationship or semantic relationship.
                                
                                We define spatial relationship if the phrase depends on the perspective, 
                                
                                i.e. left/right/front/behind and any other similar case.
                                
                                and we define semantic relationship if the phrase don't depend on the perspective,
                                
                                i.e. on/under/in or more general the phrase is a verb to describe the behaviour.
                                
                                Answer in one word, either spatial or semantic.
                                
                                The input relationship phrase: '''{relationship}'''
                                """,
                }
            ],
            model="llama3-70b-8192",
            temperature=0  # Set temperature to 0
        )
        response = chat_completion.choices[0].message.content

        return response

    except GroqError as e:
        print(f"An error occurred: {e}")
        return None


def relationship_match(relationship, pre_defined=pre_defined_rels):
    client = Groq(api_key=api_key)
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are a helpful assistant to deal with language problem. 
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                Find the word that best matches the input phrase in the pre-defined word_list.
                                
                                Only output the word you find in the pre-defined word_list.
                                
                                The pre_defined word_list: '''{pre_defined}'''

                                The input relationship phrase: '''{relationship}'''
                                """,
                }
            ],
            model="llama3-70b-8192",
            temperature=0  # Set temperature to 0
        )
        response = chat_completion.choices[0].message.content

        return response

    except GroqError as e:
        print(f"An error occurred: {e}")
        return None


def relationship_detect(query, relationship, box_sub, box_obj):
    client = Groq(api_key=api_key)
    try:
        initial_chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are a helpful assistant to analysis the if the spatial relationship exists or not. 
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                Given the relationship phrase and the bounding_boxes of subject and object, 
                                you need to judge if the relationship exists or not by comparing the bounding_boxes,
                                e.g. if the phrase describes the relative position, compare the center coordinates
                                     if the phrase describes the relative size, compare the area of bboxes
                                
                                The input include:
                                - the query is '''{query}'''
                                - the relationship phrase '''{relationship}'''
                                - the bounding_box of the subject '''{box_sub}'''
                                - the bounding_box of the object '''{box_obj}'''
                                
                                The bounding_box is in the format: [object_name, bounding_box[x,y,w,h]]
                                     
                                Let's break the task into following steps:
                                - first, analysis what kind of relationship you need to detect based on query and phrase
                                - then, choose to compare the coordinates of bboxes or the area of bboxes
                                - then, analysis weather the result satisfies the query
                                
                                Give the final output
                                - if the query is true, output 'True' as well as the query.
                                - otherwise, output 'False' as well as the correct relationship description.
                                """,
                }
            ],
            model="llama3-70b-8192",
            temperature=0  # Set temperature to 0
        )
        analysis = initial_chat_completion.choices[0].message.content

        final_chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": f"""
                                You are a helpful assistant to extract the final output from the analysis. 
                                """
                },
                {
                    "role": "user",
                    "content": f"""
                                Based on the analysis '''{analysis}'''
                                
                                Output the final result in one sentence, remove the intermediate analysis process.
                                """,
                }
            ],
            model="llama3-70b-8192",
            temperature=0  # Set temperature to 0
        )
        response = final_chat_completion.choices[0].message.content

        return response

    except GroqError as e:
        print(f"An error occurred: {e}")
        return None


if __name__ == "__main__":
    # Example sentence to decompose
    sentence = "A blue ball is within the red ball"
    subject, object_, relationship = sentence_decompose(sentence)
    print(f"Subject: {subject}")
    print(f"Object: {object_}")
    print(f"Relationship: {relationship}")
    rel = relationship_analysis(relationship)
    print(f"Relationship type: {rel}")
    rel_det = relationship_match(relationship)
    print(f"Relationship_match: {rel_det}")

