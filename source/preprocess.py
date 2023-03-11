#import nltk



#Tokenizer function. You can add here different preprocesses.
def preprocess(sentence, labels, mode=1):
	'''
	Task: Given a sentence apply all the required preprocessing steps
	to compute train our classifier, such as sentence splitting, 
	tokenization or sentence splitting.

	Input: Sentence in string format
	Output: Preprocessed sentence either as a list or a string
	'''
	# Place your code here
	# Keep in mind that sentence splitting affectes the number of sentences
	# and therefore, you should replicate labels to match.

	print (f"Using preprocessing mode {mode}")
	
	if mode == 0:
		# We notice that Chinees and Japanees could not be splitting words properly:
		for i in sentence.index:
			s = sentence.loc[i]
			if len(s.split(" ")) < 10: # Potentialy chinesse or japanese
				s_set = set(s)
				if len(s_set) > 50:
					print(labels.loc[i])
					sentence.loc[i] = " ".join(list(s))
					print(" ".join(list(s)))
	
	if mode == 1:
		# We notice that Chinees and Japanees could not be splitting words properly:
		print(sentence.index)
		print(labels.index)
		for i in sentence.index:
			#print(sentence.loc[i],"\n",labels.loc[i])
			s = sentence.loc[i]
			space_words = s.split(" ") # Num of "words"
			s_set = set(s) # Num of unique chars
			if len(space_words) < 10 or len(s_set) > 70:
				# Potentialy chinesse or japanese
				words = [char for char in s if char != " "]
				if labels.loc[i] != "Chinese" and labels.loc[i] != "Japanese": print(labels.loc[i])
				# Use N-Grams
				n=2
				ngrams=[]
				for j in range(len(words) -n + 1):
					ngrams.append("".join(words[j:j+n]))

				#words = ngrams
				sen = " ".join(ngrams)
			else:
				if labels.loc[i] == "Chinese" or labels.loc[i] == "Japanese": print("WARNING: It was "+labels.loc[i], len(space_words), len(s_set))

				#words = space_words
				sen = " ".join(space_words)

			
			#sentence.loc[i] = words
			sentence.loc[i] = sen

	return sentence,labels



