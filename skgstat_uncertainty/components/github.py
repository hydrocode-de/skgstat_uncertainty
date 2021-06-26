from github import Github

from ..core import Project


class RepoHandler:
    def __init__(self, project: Project, login_or_token: str, password: str = None):
        self.project = project

        # create github instance
        if password is None or password == '':
            self.g = Github(login_or_token)
        else:
            self.g = Github(login_or_token, password=password)
        
        # get the user
        self.user = self.g.get_user()