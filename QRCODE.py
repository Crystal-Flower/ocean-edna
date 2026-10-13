import qrcode

# The data you want to encode in the QR code (e.g., a URL, text)
data_to_encode = "https://claude.ai/public/artifacts/a70de388-5a12-4757-bbe9-e6f00e1c3fb3"

# Generate the QR code
# 'version' controls the size of the QR code (1-40)
# 'box_size' controls the size of each box in pixels
# 'border' controls the thickness of the border (in boxes)
qr_img = qrcode.make(data_to_encode)

# Save the QR code as a PNG file
file_name = "my_qr_code.png"
qr_img.save(file_name)

print(f"QR code for '{data_to_encode}' saved as '{file_name}'")