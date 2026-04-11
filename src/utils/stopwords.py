STOPWORDS = {
    # articles / prepositions
    "a", "an", "and", "are", "at", "by", "for", "from", "in", "is",
    "of", "on", "or", "the", "to", "with",
    # pronouns
    "i", "me", "my", "its", "it",
    # common conversational filler (never in vehicle records)
    "have", "has", "do", "does", "can", "could", "would", "should",
    "get", "give", "know", "tell", "show", "please",
    "want", "need", "looking", "find", "suggest", "recommend", "recommended",
    "which", "what", "how", "who",
    # tyre-query filler words (intent words, not record fields)
    "tyre", "tyres", "tire", "tires",
    "size", "version", "variant", "model",
    "right", "best", "good", "first", "second",
    "fit", "fits", "fitted", "suitable",
    "use", "using", "used",
    # common English words that cause spurious model matches (e.g. "one" → Force One)
    "one", "also", "more", "about", "just", "like", "than",
}
