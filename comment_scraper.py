'''
Scraping Data Using API from googleapis
'''

import googleapiclient.discovery
import pandas as pd
class comment_scraper():
    def __init__(self, videoId):
        self.videoid = videoId 
    
        api_service_name = "youtube"
        api_version = "v3"
        DEVELOPER_KEY = "AIzaSyCO5jsP67a4Zf_iaZTosR3zixrZMvDxlOE"

        youtube = googleapiclient.discovery.build(
            api_service_name, api_version, developerKey=DEVELOPER_KEY)

        request = youtube.commentThreads().list(
            part="snippet",
            videoId=self.videoid,
            maxResults=100
        )

        comments = []

        # Execute the request.
        response = request.execute()

        # Get the comments from the response.
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            public = item['snippet']['isPublic']
            comments.append([
                comment['authorDisplayName'],
                comment['publishedAt'],
                comment['likeCount'],
                comment['textOriginal'],
                public
            ])

        while (1 == 1):
            try:
                nextPageToken = response['nextPageToken']
            except KeyError:
                break
            nextPageToken = response['nextPageToken']
            nextRequest = youtube.commentThreads().list(part="snippet", videoId=self.videoid, maxResults=100, pageToken=nextPageToken)
            response = nextRequest.execute()
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                public = item['snippet']['isPublic']
                comments.append([
                    comment['authorDisplayName'],
                    comment['publishedAt'],
                    comment['likeCount'],
                    comment['textOriginal'],
                    public
                ])
            df = pd.DataFrame(comments, columns=['author', 'updated_at', 'like_count', 'text','public'])
            df.to_pickle("youtube_comments_df.pkl")