# Context Windows 
## .. an important property of LLMs.
The context window is telling you the total number of tokens that an LLM can examine at any one point when it's trying to generate the next token, which is its big job.

![image](https://github.com/user-attachments/assets/b87941e3-a8c9-4476-9e3f-f820187f75f1)

Now when you have a chat with something like ChatGPT, you pass in some input and it then produces some output, and then you might ask another question. Now in practice, it appears that ChatGPT seems to have some kind of a memory of what you're talking

about. It maintains context between your discussion threads, but this is something of an illusion. It's a bit of a conjuring trick. 

What's really happening is that every single time that you talk to ChatGPT the entire conversation so far, the user prompts the inputs and its responses are passed in again, as are the long prompt.

And then it ends with okay, what is most likely to come next given all of this conversation so far? So what the context window is telling you is that this is the total amount of tokens.

So the context window then is the total of all of the conversations so far, the inputs and the subsequent conversation up until the next token that it's predicting.

### This is particularly important in things like multishot prompting
