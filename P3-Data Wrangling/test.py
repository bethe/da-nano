def istobi(value):
    if str(value) == "tobi":
        print value, True
        return True
    else:
        print value, False
        return False


istobi(123)
istobi('Tobi')
istobi('tobi')
