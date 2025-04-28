import urllib.parse

raw_username = "rainyseasonof2021" # Replace with your actual username
raw_password = "p?%(j59A+VG@dh?" # Replace with your actual password (especially if it has special characters!)

encoded_username = urllib.parse.quote_plus(raw_username)
encoded_password = urllib.parse.quote_plus(raw_password)

# Now construct the URI string using these encoded parts
# Get the part of your URI after mongodb+srv:// and before the username
# It looks like it might involve %(j59a+vg@dh?@cluster0.7qqhskl.mongodb.net
# This looks suspiciously like it might be part of your hostname or a mistyped credential part.
# A standard Atlas SRV URI should look like:
# mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0

# **Double check your hostname part.** It should be something like cluster0.7qqhskl.mongodb.net.
# The warning "Unknown option: %(j59a+vg@dh?@cluster0.7qqhskl.mongodb.net/?retrywrites" is very suspicious.
# It suggests you might have accidentally included extra characters or an incorrect format *before* the @ symbol that separates credentials from the hostname.

# Let's assume your *actual* hostname part is `cluster0.7qqhskl.mongodb.net`.
hostname_part = "cluster0.7qqhskl.mongodb.net" # Replace if different
options_part = "?retryWrites=true&w=majority&appName=Cluster0" # Replace if different

encoded_uri = f"mongodb+srv://{encoded_username}:{encoded_password}@{hostname_part}/{options_part}"

print("Use this encoded URI in your .env:")
print(encoded_uri)