# """
# Get Retrieved Context Query - Truy vấn lấy context từ vector database

# File này định nghĩa GetRetrievedContextQuery để lấy context từ vector database.
# """

# from dataclasses import dataclass
# from core.cqrs import Query


# @dataclass
# class GetRetrievedContextQuery(Query):
#     """
#     Query lấy context từ vector database

#     Attributes:
#         query (str): Query cần lấy context
#     """

#     query: str
