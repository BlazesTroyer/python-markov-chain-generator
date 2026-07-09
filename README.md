# python-markov-chain-generator
A simple, 1st order Markov Chain text generator I built to understand the basics of language models.

Interactive console app that uses a Markov Chain algorithm to learn from the text you feed it and generate new sentences. 

This project relies purely on Python's built in libraries `random`, `json`, and `os`.

Features

Learns as you type: Analyzes your input to understand how words connect and builds its own vocabulary.
Persistent Memory: Automatically saves everything it learns to a `chain_memory.json` file, so the model remembers its vocabulary even after you close the app.
Auto Recovery: Checks for the memory file on startup; if it's corrupted or missing, it gracefully starts fresh from zero.
Flexible Generation: You can either give it a specific starting word and sentence length, or just let it pick a random word and create entirely on its own.
Auto Formatting: Cleans up the generated text by capitalizing the first words of sentences and adding proper punctuation ".", "?", "!" at the end.

How to Run

1. Download this repository.
2. Open your terminal and navigate to the folder containing the script.
3. Run the script.
