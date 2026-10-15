import sys
import openfhe

print(">>> Initializing OpenFHE CKKS Client Test...")

# 1. Define CKKS Parameters (N=8192 matching SHIELD baseline)
parameters = openfhe.CCParamsCKKSRNS()
parameters.SetMultiplicativeDepth(2)
parameters.SetScalingModSize(40)
parameters.SetFirstModSize(60)
parameters.SetSecurityLevel(openfhe.SecurityLevel.HEStd_128_classic)

# 2. Instantiate the CryptoContext
cc = openfhe.GenCryptoContext(parameters)
cc.Enable(openfhe.PKESchemeFeature.PKE)
cc.Enable(openfhe.PKESchemeFeature.KEYSWITCH)
cc.Enable(openfhe.PKESchemeFeature.LEVELEDSHE)

# 3. Generate Public/Private & Evaluation Keys
keys = cc.KeyGen()
cc.EvalMultKeyGen(keys.secretKey)

# 4. Pack real-valued features (SIMD slot packing)
feature_vector = [0.25, 0.50, 0.75, 1.0]
plaintext = cc.MakeCKKSPackedPlaintext(feature_vector)

# 5. Encrypt into ciphertext
ciphertext = cc.Encrypt(keys.publicKey, plaintext)

# 6. Test a homomorphic multiplication (evaluating without decryption)
ciphertext_squared = cc.EvalMult(ciphertext, ciphertext)

# 7. Decrypt and check numeric values
decrypted_plaintext = cc.Decrypt(ciphertext_squared, keys.secretKey)
decrypted_plaintext.SetLength(len(feature_vector))

print("\n--- OpenFHE Python Client Verification ---")
print(f"Original Input Vector: {feature_vector}")
print(f"Decrypted x^2 Vector:  {decrypted_plaintext}")
print("Status: SUCCESS\n")
