## Understanding LLM Parameters: from GPT-1 to Trillion-Weight Models

![image](https://github.com/user-attachments/assets/97d32d28-d646-4792-8dcc-6807b3554710)

## Tokens:
Are the individual units which get passed into a model. Diffeent tokenizers work differently. There are pros and cons for having fewer or more tokens. checkout https://platform.openai.com/tokenizer  the gap between words is included as part of a token

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

## Training Neural Networks later on..
Models were trained on each individual possible word. It was much easier to learn from but leaded to enormous vocab with rare omitted words (you have places and names also as words)

## Breakthrough:
 rather than trying to train a model based on individual characters and need it to learn how to combine them to form a word. and rather than trying to say that each word is a different token, 

you could take chunks of letters, chunks that would that would sometimes form a complete word and sometimes part of a word, and call it a token, and train the model to take a series of tokens and output tokens based on the tokens that are passed in.

And this had a number of interesting benefits. One of them is that because you're breaking things down into tokens, you could also handle things like names of places and proper names.

They would just be more fragments of tokens. And then there was a second interesting effect, which is that it meant that it was good at handling word stems, or times when you'd have the same beginning of a word and multiple potential endings that would be encoded into one token, followed by a few second tokens. And that meant that the sort of underlying meaning of what you're trying to say could be easily represented inside the model, because the tokens had the same kind of structure that might have sounded a bit abstract.
