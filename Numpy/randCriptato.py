from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from os import urandom

class FortunaRNG:
    def __init__(self, seed=None):
        self.key = urandom(32)
        self.counter = 0
        self.seed(seed)

    def seed(self, seed):
        if seed is not None:
            self.key = self._hash(self.key + seed)

    def _hash(self, data):
        from hashlib import sha256
        return sha256(data).digest()

    def _encrypt(self, plaintext):
        cipher = Cipher(algorithms.AES(self.key), modes.ECB(), backend=default_backend())
        encryptor = cipher.encryptor()
        return encryptor.update(plaintext) + encryptor.finalize()

    def random(self, n):
        self.counter += 1
        counter_bytes = self.counter.to_bytes(16, byteorder='big')
        random_bytes = self._encrypt(counter_bytes)[:n]
        return int.from_bytes(random_bytes, byteorder='big')

    def randint(self, min_value, max_value):
        range_size = max_value - min_value + 1
        num_bits = range_size.bit_length()
        num_bytes = (num_bits + 7) // 8

        while True:
            random_value = self.random(num_bytes)
            if random_value < (1 << num_bits) - ((1 << num_bits) % range_size):
                break

        return (random_value % range_size) + min_value

# Esempio d'uso:
rng = FortunaRNG()

min_value = 1
max_value = 6
random_number = rng.randint(min_value, max_value)
print(f"Numero casuale tra {min_value} e {max_value}: {random_number}")
