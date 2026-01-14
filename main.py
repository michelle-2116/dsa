from memory import MainHistory, LLMContext
from pinecone_utils import upsert_message, query_similar_turns
from llm import ask_llm

history = MainHistory()
context = LLMContext()

while True:
    user_input = input("You: ").strip()

    if user_input.lower() in ["exit", "quit", "q"]:
        print("Exiting chat")
        break

    relevant_turn_ids = query_similar_turns(user_input, threshold=0.40) #retrieve relevant from pinecone
    
    context.clear()
    added_turns = set() #prevent duplicates from being sent to context

    for tid in relevant_turn_ids:
        if tid < len(history.history):
            context.add(history.history[tid])
            added_turns.add(tid)

    last_idx = len(history.history) - 1 #get last turn
    if last_idx >= 0 and last_idx not in added_turns:
        context.add(history.history[last_idx])

    history_prompt = context.to_prompt()
    full_prompt = history_prompt + f"User: {user_input}\nAssistant: "
    
    reply = ask_llm(full_prompt)
    print("LLM:", reply)

    current_turn_id = history.add_turn(user_input, reply)

    upsert_message(f"{current_turn_id}_u", user_input, current_turn_id, "user")
    upsert_message(f"{current_turn_id}_l", reply, current_turn_id, "llm")

    print("\n---promptsent---")
    print(full_prompt)
    print("-------------\n")