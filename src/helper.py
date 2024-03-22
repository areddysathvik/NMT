import tensorflow as tf
import pickle
from keras.utils import pad_sequences
from numpy import argmax,zeros

encoder_model = tf.keras.models.load_model(r'C:\Users\aredd\Desktop\NMT\MODELS\model_prd_enc_tf')
decoder_model = tf.keras.models.load_model(r'C:\Users\aredd\Desktop\NMT\MODELS\model_prd_dec_tf')

with open(r'C:\Users\aredd\Desktop\NMT\Tokenizers\tokenizer.pickle', 'rb') as handle:
    input_tokenizer = pickle.load(handle)

with open(r'C:\Users\aredd\Desktop\NMT\Tokenizers\tokenizer_fr.pickle', 'rb') as handle:
    output_tokenizer = pickle.load(handle)

inputs_word2index = input_tokenizer.word_index
outputs_word2index = output_tokenizer.word_index

index_to_word_input = {v:k for k,v in inputs_word2index.items()}
index_to_word_output = {v:k for k,v in outputs_word2index.items()}
inputs_maxlen = 15
outputs_maxlen = 22


def translate_to_FRENCH(input_en):
    input_seq = input_tokenizer.texts_to_sequences([input_en])
    encoder_input_sequences = pad_sequences(input_seq, maxlen=inputs_maxlen)
    def translate(input_seq):
        states = encoder_model.predict(input_seq,verbose=0)
        
        sos = outputs_word2index['<sos>']
        eos = outputs_word2index['<eos>']
        
        output_seq = zeros((1, 1))
        output_seq[0, 0] = sos
        
        output_sentence = []
        
        for _ in range(outputs_maxlen):
            output_tokens, h, c = decoder_model.predict([output_seq]+states,verbose=0)
            idx = argmax(output_tokens[0, 0, :])
            
            if idx == eos:
                break     
            word = ''
            if idx > 0:
                word = index_to_word_output[idx]
                output_sentence.append(word)
            
            states = [h, c]
            output_seq[0, 0] = idx
        
        return ' '.join(output_sentence)
    
    translation = translate(encoder_input_sequences)
    return translation