def get_conversation_string(st):
    conversation_string = ""
    for i in range(len(st.session_state['responses'])-1):        
        conversation_string += "Human: "+st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: "+ st.session_state['responses'][i+1] + "\n"
    return conversation_string

def get_relevant_documents(collection, query, k=2):
    results = collection.query(
        query_texts=[query],
        n_results=k
    )
    return results['documents'][0][0]+"\n\n"+results['documents'][0][1]
