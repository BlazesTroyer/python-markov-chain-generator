import random
import json
import os

memory = {}
file_name = "chain_memory.json" # We named our memory file here. We are going to save our memory to a file so it is going to remember everything we taught it.

def save_memory():
    """
    Saves the current state of the memory to JSON file.
    """
    with open(file_name, "w", encoding = "utf-8") as file: 
        json.dump(memory, file, ensure_ascii = False, indent = 4)

    print(f"Model's current memory has been saved to {file_name} successfully.")

def load_memory():
    """
    Loads the model's memory from the JSON file if it exists.
    If the file is missing or corrupted, creates a new one instead.
    """
    global memory

    if os.path.exists(file_name): # Checking if we already had a file which named as same as we were going to create, so if we already had it, it will try to load it firstly instead of creating a new one.
        try:
            with open(file_name, "r", encoding = "utf-8") as file: # Checking the file we got.
                memory = json.load(file) # Loading everything inside it to memory.

            print(f"The old memory file has been found successfully! Total {len(memory)} words are known.")

        except json.decoder.JSONDecodeError: # Checking if the file is corrupted, that would cause an error. That is why we use "try" and "except".
            memory = {} 
            print(f"Memory file was corrupted, starting from zero.")

    else:
        print(f"The {file_name} could not be found, model is getting started from zero.") # If there is not a file which named with our settled file name, it going to create one.

def clean(text):
    """
    Prepares the input text for process.
    """
    cleantext = text.replace(".", " . ").replace("?", " ? ").replace("!", " ! ") # Putting a space between last words and "."s so we can take words by only themselves.
    words = cleantext.lower().split() # Splitting the text's words and converting its letters to lowercase one by one.
    return words
    
def start(words):
    """
    Builds the markov chain memory.
    """
    for i in range(len(words) - 1): # It is going to render words with the next words instead of only itself, so we have to start from one behind. Or, we are going to check for one more word which is not here.
        cur_word = words[i]
        next_word = words[i + 1]
        
        if cur_word in [".", "?", "!"]: # If the current word is ".", "?", "!" just passes instead of looking at the next one for learning. Because next word will be either nexts sentence's first word, or either nothing.
            continue    

        if cur_word not in memory: # Adding the current word into the memory if it is not already there.
            memory[cur_word] = []
            
        memory[cur_word].append(next_word) # Adding the next word under the current word.
        
def learn(): # Learning codes are going to work with this def.
    """
    Takes user's input, saves it to model's memory
    after cleaning it.
    """
    text = input("Enter your text to make your model learn something: ")
    if not text.strip(): # If you only press "enter", this code will prevent crash.
        return
    
    words = clean(text)
    start(words)
    save_memory()  

def generate(firstword, length): # Going to generate a sentence by using first word and the length of sentence.
    """
    Generates a sentence by using markov chain algorithm.
    """
    if firstword not in memory: # It cannot create a sentence with a word which it does not know afterall.
        print(f"\n'{firstword}' is not in the memory of model yet.") 
        return

    generated_words = [firstword] # Adding our first word to generated words, so it can generate next one by looking at the current one.
    
    curlength = 0 # Current sentence length (words).
    while length - curlength > 1: # Repeats until current sentence length catches the requirement sentence length.
        curword = generated_words[-1]        
        
        if curword in memory and len(memory[curword]) > 0: 
            nextword = random.choice(memory[curword])

            if nextword in [".", "?", "!"]:
                curlength -= 1

        else: # If the chosen word has not any words under it, sentence cannot continue.
            nextword = random.choice(list(memory.keys())) # So we are going to choose an other random word for that.
            # In older version, I used to use "break" instead of random. Because if you use "break", it will directly end the sentence and put "." end of it.                
            # But the problem here is, even if you told it to create a sentence with six words, it can end that sentence at one word.
            # If you use random instead, the sentence's meaning probably going to break down but continue at least.

        generated_words.append(nextword) # Adding our selected next word to our sentence (generated_words).
        curlength += 1

    sentence = ""
    capitalize_next = True 

    for word in generated_words:
        if word in [".", "?", "!"]: # Checking if the current word is one of ".", "?", "!".
            sentence = sentence.strip() + word + " "
            capitalize_next = True # We capitalise the words that come after ".", "?", "!" here.

        elif capitalize_next:
            sentence += word.capitalize() + " "
            capitalize_next = False
            
        else:
            sentence += word + " " # Just continues to building sentence normally if there is not any ".", "?", "!" found yet.

    sentence = sentence.strip() 
    if not sentence.endswith((".", "?", "!")): # Checking if sentence has ".", "?", "!" at the end of it.
        sentence += "." # Putting "." at end of the sentence.
        sentence = sentence.replace(" .", ".")

    print(f"\n{sentence}\n")

def speak():
    """
    Allows the user either give a starting word and length for generating a sentence
    or just lets the model handle it by itself.
    """
    while True:
        answer = input("Should model speak by itself or would you want to give it command? \n(itself / me): ").lower().strip()
        if answer == "me":
            firstword = input("What word should sentence start with?: ").lower().strip()
            try:
                length = int(input("How many words should be in the sentence?: "))
                generate(firstword, length)
                break

            except ValueError:
                print("Please enter a valid number for sentence length.")

        elif answer == "itself":
            if not memory:
                print("Memory is empty right now.")
                break

            firstword = random.choice(list(memory.keys())) # Going to select its own first word by randomly.
            length = random.randint(3, 10)
            generate(firstword, length)  
            break

        else:
            print("Please give a valid answer.")

load_memory() # Before starting to take commands, tries to settle its memory.

while True:
    choice = input("Would you want to teach something new to model or make it speak instead? \n(t - teach / s - speak / q - quit): ").lower().strip()
    if choice == "q":
        exit()

    elif choice == "t":
        learn()

    elif choice == "s":
        if not memory:
            print("Sadly model does not know anything right now. You have to teach it something before making it speak.")
            learn()
        
        else:
            speak()

    else:
        print("Please give a valid answer.")