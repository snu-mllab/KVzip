"""
This script is adapted from 
https://github.com/FranxYao/Long-Context-Data-Engineering
"""

import glob
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)


class NeedleHaystackData:

    def __init__(
        self,
        tokenizer,
        context_lengths=[8000],
        haystack_dir="/home/janghyun/Codes/memory/data/needle/PaulGrahamEssays",
        needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        retrieval_question="Based on the content of the book, the best thing to do in San Francisco is: ",
        final_context_length_buffer=200,
        model_provider="LLaMA3",
    ):
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.enc = tokenizer

        self.context_lengths = context_lengths
        self.final_context_length_buffer = final_context_length_buffer
        self.model_provider = model_provider
        self.retrieval_question = retrieval_question

    def read_context_files(self):
        context = ""
        max_context_length = max(self.context_lengths)
        file_list = glob.glob(f"{self.haystack_dir}/*.txt")
        if len(file_list) == 0:
            raise FileNotFoundError(f"No files found in {self.haystack_dir}")

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in file_list:
                with open(file, 'r') as f:
                    context += f.read()

        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context, add_special_tokens=False)

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)  # Changed from True to False

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length])

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))

    def insert_needle(self, context, depth_percent, context_length):
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
        context_length -= self.final_context_length_buffer

        # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[:context_length - len(tokens_needle)]

        if depth_percent == 100:
            # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
            tokens_new_context = tokens_context + tokens_needle
            print("insertion at %d" % len(tokens_context))
        else:
            # Go get the position (in terms of tokens) to insert your needle
            insertion_point = int(len(tokens_context) * (depth_percent / 100))

            # tokens_new_context represents the tokens before the needle
            tokens_new_context = tokens_context[:insertion_point]

            # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
            period_tokens = self.encode_text_to_tokens('.') + self.encode_text_to_tokens('.\n')

            # Then we iteration backwards until we find the first period
            while tokens_new_context and tokens_new_context[-1] not in period_tokens:
                insertion_point -= 1
                tokens_new_context = tokens_context[:insertion_point]

            print("insertion at %d" % insertion_point)
            # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
            # Now we have a needle in a haystack
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # Convert back to a string and return it
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def generate_context(self, context_length, depth_percent):
        context = self.read_context_files()
        context = self.encode_and_trim(context, context_length)
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def generate_prompt(self, context):
        # Replace the following line with the appropriate prompt structure
        test_format = f"<|begin_of_text|> This is a very long story book: {context} <|eot_id|>.\nQuestion: {self.retrieval_question}\nAnswer:"
        return test_format

    def generate_context_qa(self, context_length, depth_percent):
        context = self.generate_context(context_length, depth_percent)

        q = self.retrieval_question
        a = "Eat a sandwich and sit in Dolores Park on a sunny day."

        return {"context": context.strip(), "question": [q], "answers": [a]}


if __name__ == "__main__":
    import argparse
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--context_length', type=int, default=8000, help='a number')
    parser.add_argument('-d', '--depth_percent', type=int, default=50, help='a number')
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    ht = NeedleHaystackData(tokenizer, [args.context_length])

    context = ht.generate_context(args.context_length, args.depth_percent)
    prompt = ht.generate_prompt(context)

    print()
    print(prompt)
