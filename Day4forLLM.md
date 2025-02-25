## Understanding LLM Parameters: from GPT-1 to Trillion-Weight Models

![image](https://github.com/user-attachments/assets/97d32d28-d646-4792-8dcc-6807b3554710)

### Tokens:
Are the individual units which get passed into a model.

## Early days- training Neural Networks
In the early days of building neural networks, one of the things that you'd see quite often is neural 
networks that were trained character by character.
So you would have a model which would take a series of individual characters, and it would be trained such that it would predict the most likely next character, given the characters that have come before. That was a particular technique, and in some ways it had a lot of benefits. It meant that the number of possible inputs was a limited number, just the number of possible letters
of the alphabet and some symbols. And so that meant that it had a very manageable vocab size.

And it needed to its weights could didn't didn't need to worry about too many different possibilities

for the inputs.

But the challenge with it was that it meant that there was so much required from the model in terms

of understanding how a series of different characters becomes a word, and all of the intelligence associated

with the meaning behind a word had to be captured within the weights of the model, and that was expecting

too much from the model itself.
