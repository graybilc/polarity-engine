def parse_fasta_content(fasta_text: str) -> dict[str, str]:
    """
    Parse a FASTA string

    Params:
        fasta_text (str): string object containing metadata and amino acid sequence for a
        protein separated by a newline character
    Returns:
        dict of the folllowing key, value pairs
            metadata (str): metadata associated with the amino acid found
            aa_sequence (str): amino acid sequence of the specified protein
    """
    if not fasta_text.strip():
        raise ValueError("Empty FASTA data received.")

    lines = fasta_text.strip().split("\n")
    header = lines[0]
    seq_string = "".join(lines[1:]).upper()
    if not header:
        logging.error("Malformed FASTA header format")
    elif not seq_string:
        logging.error("Amino acid sequence is empty.")
    else:
        return {"metadata": header, "aa_sequence": seq_string}