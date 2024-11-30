from crewai import Agent
from crewai_tools import WebsiteSearchTool, SerperDevTool, FileReadTool

web_search_tool = WebsiteSearchTool()
serper_dev_tool = SerperDevTool()
file_read_tool = FileReadTool(
    file_path="job_description_example.md",
    description="A tool to read the job description example file"
)


class Agents():
    def research_agent(self):
        return Agent(
            role="Research Analyst",
            goal="Analyze the company website and provided description to extract insights on culture, values and specific needs",
            tools=[web_search_tool, serper_dev_tool],
            backstory="Expert in analyzing company cultures and identifying  key values and needs from various sources including websites and brief descriptions",
            verbose=True
        )

    def writer_agent(self):
        return Agent(
            role="Job Description Writer",
            goal="Use insights from the research analyst to create a detailed, engaging and awesome job posting",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="Skilled in crafting a compelling job description that outstand or resonate with the company values and attract the right candidates",
            verbose=True
        )

    def review_agent(self):
        return Agent(
            role="Review and Editing Specialist",
            goal="Review the job posting for clarify, engagement, grammatical accuracy and alignment with company values and refine it to ensure it's perfect",
            tools=[web_search_tool, serper_dev_tool, file_read_tool],
            backstory="A strict and  meticulous editor with an eye for detail, ensuring every piece of content is clear, engaging and gramatically perfect",
            verbose=True
        )