from rich.console import Console

console = Console()

def error(title,message,Environment_Information=None):
    if len(title)>=57:
        line='-'*3
    else:
        line= '-'*(57-len(title))
    console.print(f"\n {title} {line}",style="yellow")
    console.print(f"\n  {message}")
    if Environment_Information is not None:
        Environment_Information=str(Environment_Information)
        Environment_Information=Environment_Information.replace('\n', '\n       ')
        console.print('\n     Your Environment Information ---------------------------',style="yellow")
        console.print(f'       {Environment_Information}',style="yellow")

if __name__ == '__main__':
    message = """An error occurred: HelloLambdaFunction - Resource handler returned message: "Value nodejs12.xa at 'runtime' failed to satisfy constraint: Member must satisfy enum value set: [nodejs12.x, python3.6, provided, nodejs14.x, ruby2.7, java11, go1.x, provided.al2, java8, java8.al2, dotnetcore3.1, python3.7, python3.8] or be a valid ARN"""
    info = """Operating System: darwin
Node Version: 15.14.0
Framework Version: 2.52.1
Plugin Version: 5.4.3
SDK Version: 4.2.5
Components Version: 3.14.0"""
    error("Serverless Error",message,info)