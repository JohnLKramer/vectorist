class CliError(Exception):
    """A fatal, user-facing CLI failure."""


class ConfigError(CliError):
    pass


class RepoError(CliError):
    pass


class ServerError(CliError):
    pass


class CrawlError(CliError):
    pass
