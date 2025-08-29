analyst_format = """
The output should be formatted as a JSON instance that conforms to the JSON schema below.

As an example, for the schema {{{{"properties"}}: {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}
the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

Here is the output schema:
```
{{"properties": {{"foo": {{"title": "foo", "description": "fooooo", "type": "string"}}, "bar": {{"title": "bar1", "description": "one of the bars", "type": "string"}}, "bar2": {{"title": "bar", "description": "one of the bars", "type": "string"}}, "baz1": {{"title": "baz1", "description": "one of the baz", "type": "string"}}, "baz2": {{"title": "baz2", "description": "one of the baz", "type": "string"}}}}, "required": ["foo", "bar1", "bar2", "baz1", "baz2"]}}
```
"""

# this should be generated using like:
# parser_analyst = JsonOutputParser(pydantic_object=Article) #article is from "data structures"
# analyst_format = parser_analyst.get_format_instructions()
# print (analyst_format)
# copy and paste
# and then escape the curly braces by doubling them