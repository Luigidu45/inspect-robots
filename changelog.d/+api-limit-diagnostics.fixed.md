**Agent plugin:** Chat and Responses distinguish recognized rate limits from credit,
spending, and organization usage errors. HTTP 429 retry notices show readable
waits and provider-supplied limits without printing raw error payloads or
promising a quota reset. Explicit billing and usage rejections stop without
retrying on both wires.
