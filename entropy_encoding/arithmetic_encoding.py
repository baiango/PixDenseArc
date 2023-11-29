from fractions import Fraction
from math import log


def count_frequency(text):
	frequency = {}
	for letter in text:
		if letter in frequency:
			frequency[letter] += 1
		else:
			frequency[letter] = 1
	return frequency


def count_distribute_probability(frequency):
	"""
	Calculates the probability ranges for each character in a message based on
	frequency.

	This function takes a dictionary of character frequencies and converts it
	into a dictionary of probability ranges for each character. These ranges
	are represented as tuples of fractions, indicating the start and end of
	each character's probability range.

	Parameters:
	frequency (dict):	A dictionary with characters as keys and their
						frequency counts as values.

	Returns:
	dict:	A dictionary with characters as keys and tuples (start, end) as
			values representing the probability range of each character.
	"""
	# Calculate the total count of all characters
	total_count = sum(frequency.values())

	# Convert frequencies to probabilities
	char_probabilities = {char: Fraction(count, total_count) for char, count in frequency.items()}

	# Initialize the dictionary for probability ranges
	probability_ranges = {}

	# Start with a cumulative probability of zero
	previous_probability = Fraction(0)

	# Calculate the cumulative probability ranges for each character
	for char, probability in char_probabilities.items():
		# Define the start of the current range
		current_range_start = previous_probability

		# Define the end of the current range
		current_range_end = previous_probability + probability

		# Store the range for the character
		probability_ranges[char] = (current_range_start, current_range_end)

		# Update the cumulative probability for the next character
		previous_probability = current_range_end

	return probability_ranges


def arithmetic_encode(message, symbol_probabilities):
	"""
	Compresses a message into a fraction using arithmetic encoding. This
	function uses fractions to determine the probability of each symbol in the
	message, aiming for precision in the encoding process.

	Optimal compression is often achieved with a moderate level of probability
	precision, approximately 1 and 1/3 times the count of unique symbols.
	Exceeding this level of precision can lead to less efficient compression
	due to the need to quantize the probabilities or because of an increased
	size of the final compressed fraction.

	A noteworthy aspect of this method is that when the denominators of the
	symbol probabilities are powers of 2, the logarithm base 2 of the
	denominator can be used. This allows for a more efficient storage,
	potentially saving the information in 6 bits instead of 8 bytes. However,
	this is specific to cases where such a power-of-2 denominator is present.

	These observations and techniques are based on a limited set of tests and
	may not apply universally. Different data types or probability
	distributions might yield different results.

	Parameters:
	message (str): The text message to be compressed.
	symbol_probabilities (dict):	A dictionary where each key is a symbol
									from the message and each value is a tuple
									representing the probability range of that
									symbol as fractions.

	Returns:
	tuple:	A tuple containing the compressed fraction, the logarithm (base 2)
			of the denominator of the fraction (if it's a power of 2), and the
			encoding process steps.

	Note:
	The effectiveness of this compression method can vary based on the input
	data. It is recommended to experiment with different data and probability
	distributions to fully understand the method's applicability and
	efficiency.

	Encoding example:
	message = 'kiwi'
	probabilities = {'k': (0, Fraction(1, 3)), 'i': (Fraction(1, 3), Fraction(2, 3)), 'w': (Fraction(2, 3), 1)}

	- 'k': Probability range from 0 to 1/3
	- 'i': Probability range from 1/3 to 2/3
	- 'w': Probability range from 2/3 to 1

	Now, let's go through the encoding process for "kiwi":
	1.	Start with the full interval [0, 1].
	2.	Encoding 'k':
			The range for 'k' is [0, 1/3]. So, our interval after encoding 'k'
			is [0, 1/3).
	3.	Encoding 'i':
			We split the current interval [0, 1/3] for 'i'. The 'i' falls in
			the middle third of any interval. So, the new interval is
			[1/9 (which is 1/3 of 0), 2/9 (which is 1/3 of 1/3)].
	4.	Encoding 'w':
			Now we split the interval [1/9, 2/9) for 'w'. The 'w' takes the
			last third of any interval. So, the new interval is
			[5/27 (which is 2/3 of 1/9), 6/27 (which is 2/3 of 2/9)].
	5.	Encoding the final 'i':
			Finally, we split the interval [5/27, 6/27) for 'i'. The 'i' takes
			the middle third. So, the final interval is
			[16/81 (which is 1/3 of 5/27), 17/81 (which is 1/3 of 6/27)].
	6.	Choose a fraction within the final interval:
			Any fraction within [16/81, 17/81) can represent "kiwi".
			For simplicity, let's take the middle of this interval:
			(16/81 + 17/81) / 2 = 33/162.

	So, the string "kiwi" can be represented by the fraction 33/162 (or 11/54)
	in this arithmetic encoding scheme.
	"""
	lower_bound = Fraction(0, 1)
	upper_bound = Fraction(1, 1)
	encoding_process = []

	for char in message:
		range_width = upper_bound - lower_bound
		upper_bound = lower_bound + range_width * symbol_probabilities[char][1]
		lower_bound = lower_bound + range_width * symbol_probabilities[char][0]
		encoding_process.append((char, lower_bound, upper_bound))

	fraction = (lower_bound + upper_bound) / 2

	return fraction, log(fraction.denominator, 2), encoding_process


def arithmetic_decode(encoded_number, symbol_probabilities, message_length):
	"""
	Decoding example:
	To decode the fraction 33/162 back into the string "kiwi" using the given
	probabilities, we follow the reverse process of arithmetic encoding.
	Here's how it would work:

	1. Start with the encoded fraction: We have the fraction 33/162.
	2. Initialize the interval: The full interval is [0, 1].
	3. Determine the symbol for each step:
	-	Check which interval the fraction 33/162 falls into based on our symbol
		probabilities:
		-	'k': [0, 1/3]
		-	'i': [1/3, 2/3]
		-	'w': [2/3, 1]

	For each step, find the symbol whose range contains the fraction:
	a.	First Symbol:
		-	33/162 falls in [0, 1/3], so the first symbol is 'k'.
	b.	Adjust the interval and fraction for the next symbol:
		-	The new interval for 'k' is [0, 1/3]. Scale the fraction to fit in
			this new interval:
				33/162, when scaled to fit in [0, 1/3],
				becomes 33/162 * 3 = 99/162.
	c.	Second Symbol:
		-	99/162 falls in [1/3, 2/3], so the second symbol is 'i'.
		-	Adjust the interval for 'i', which is [1/3, 2/3].
			Scale the fraction to fit in this interval:
				([99/162] - 1/3) * 3 = 33/162.
	d.	Third Symbol:
		-	33/162 falls in [2/3, 1], so the third symbol is 'w'.
		-	Adjust the interval for 'w', which is [2/3, 1].
			Scale the fraction to fit in this interval:
				([33/162] - 2/3) * 3 = 33/162.
	e.	Fourth Symbol:
		-	33/162 falls in [1/3, 2/3], so the fourth symbol is 'i'.

	4.	Combine the symbols:
		We've determined the sequence of symbols to be "kiwi".
		So, by decoding the fraction 33/162 using the given probabilities,
		we retrieve the original string "kiwi". This decoding process involves
		determining which symbol's range contains the fraction at each step and
		then scaling and shifting the fraction to prepare for the next step.
	"""
	lower_bound = Fraction(0, 1)
	upper_bound = Fraction(1, 1)
	decoded_message = []
	for _ in range(message_length):
		range_width = upper_bound - lower_bound
		for symbol, (symbol_lower, symbol_upper) in symbol_probabilities.items():
			if lower_bound <= encoded_number < lower_bound + range_width * symbol_upper:
				decoded_message.append(symbol)
				upper_bound = lower_bound + range_width * symbol_upper
				lower_bound = lower_bound + range_width * symbol_lower
				break
	return decoded_message


if __name__ == '__main__':
	def arithmetic_encode_and_verify(message, symbol_probabilities):
		encoded_value, logarithm, encoding_process = arithmetic_encode(message, symbol_probabilities)
		decoded_message = arithmetic_decode(encoded_value, symbol_probabilities, len(message))
		is_correct = decoded_message == list(message)

		return encoded_value, logarithm, is_correct, encoding_process

	message = 'kiwikiwikiwikiwikiwiiiiwwwwkikikiikwikwikw'
	# message = 'kiwi' # Docstring example 1
	probabilities = {'k': (0, Fraction(1, 3)), 'i': (Fraction(1, 3), Fraction(2, 3)), 'w': (Fraction(2, 3), 1)}
	print(probabilities)
	encoded_value, logarithm, is_correct, encoding_process = arithmetic_encode_and_verify(message, probabilities)
	# print(f"Process: {encoding_process}") # Print here
	print(f"Encoded Value: {encoded_value}")
	print(f"logarithm of the denominator: {logarithm}")
	print(f"Decoding Correct: {is_correct}")

	probabilities = count_distribute_probability(count_frequency(message))
	print(probabilities)
	encoded_value, logarithm, is_correct, encoding_process = arithmetic_encode_and_verify(message, probabilities)
	print(f"Encoded Value: {encoded_value}")
	print(f"logarithm of the denominator: {logarithm}")
	print(f"Decoding Correct: {is_correct}")

	probabilities = {'k': (0, Fraction(2, 8)), 'i': (Fraction(2, 8), Fraction(5, 8)), 'w': (Fraction(5, 8), 1)}
	print(probabilities)
	encoded_value, logarithm, is_correct, encoding_process = arithmetic_encode_and_verify(message, probabilities)
	print(f"Encoded Value: {encoded_value}")
	print(f"logarithm of the denominator: {logarithm}")
	print(f"Decoding Correct: {is_correct}")

	probabilities = {'k': (0, Fraction(1, 4)), 'i': (Fraction(1, 4), Fraction(3, 4)), 'w': (Fraction(3, 4), 1)}
	print(probabilities)
	encoded_value, logarithm, is_correct, encoding_process = arithmetic_encode_and_verify(message, probabilities)
	print(f"Encoded Value: {encoded_value}")
	print(f"logarithm of the denominator: {logarithm}")
	print(f"Decoding Correct: {is_correct}")
