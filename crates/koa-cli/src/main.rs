use std::env;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use koa_core::{ChatOptions, KoaRuntime};
use serde_json::json;

#[derive(Debug, Parser)]
#[command(name = "koa", version, about = "Koa native agentic runtime")]
struct Cli {
    #[arg(long, global = true)]
    root: Option<PathBuf>,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Init,
    Doctor,
    Chat(ChatArgs),
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Capsule {
        #[command(subcommand)]
        command: CapsuleCommand,
    },
    Session {
        #[command(subcommand)]
        command: SessionCommand,
    },
    Vault {
        #[command(subcommand)]
        command: VaultCommand,
    },
    Tools {
        #[command(subcommand)]
        command: ToolsCommand,
    },
    Skill {
        #[command(subcommand)]
        command: SkillCommand,
    },
    Agent {
        #[command(subcommand)]
        command: AgentCommand,
    },
}

#[derive(Debug, Args)]
struct ChatArgs {
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 0.7)]
    temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    prompt: Vec<String>,
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    Doctor,
    Verify,
    Manifest {
        #[arg(long)]
        revision: String,
        #[arg(long = "file")]
        files: Vec<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum CapsuleCommand {
    Doctor,
}

#[derive(Debug, Subcommand)]
enum SessionCommand {
    Create { name: String },
    List,
    Switch { id: String },
    Snapshot { name: String },
    Restore { name: String },
    Destroy { id: String },
}

#[derive(Debug, Subcommand)]
enum VaultCommand {
    Ingest {
        path: PathBuf,
    },
    Search {
        query: String,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    Lint,
}

#[derive(Debug, Subcommand)]
enum ToolsCommand {
    List,
    Register {
        manifest: PathBuf,
    },
    Call {
        id: String,
        #[arg(long, default_value = "{}")]
        input: String,
    },
    RegisterMcp {
        name: String,
        command: String,
    },
}

#[derive(Debug, Subcommand)]
enum SkillCommand {
    Create {
        id: String,
        name: String,
        description: String,
        body: String,
        #[arg(long)]
        global: bool,
    },
    List,
    Delete {
        id: String,
    },
}

#[derive(Debug, Subcommand)]
enum AgentCommand {
    Spawn {
        id: String,
        role: String,
        goal: String,
        #[arg(
            long,
            default_value = "result.summary, key_findings, risks_or_uncertainties, required_next_actions"
        )]
        output_contract: String,
        #[arg(long)]
        global: bool,
    },
    List,
    Delete {
        id: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = cli.root.unwrap_or(env::current_dir()?);

    match cli.command {
        Command::Init => {
            let runtime = KoaRuntime::init(&root)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "status": "initialized",
                    "root": runtime.paths().root,
                    "state": runtime.paths().state,
                }))?
            );
        }
        Command::Doctor => {
            let runtime = KoaRuntime::open(&root)?;
            let report = runtime.doctor();
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.healthy() {
                std::process::exit(2);
            }
        }
        Command::Chat(args) => {
            let runtime = KoaRuntime::open(&root)?;
            let prompt = args.prompt.join(" ");
            if prompt.trim().is_empty() {
                anyhow::bail!("prompt cannot be empty");
            }
            println!(
                "{}",
                runtime.chat_with_options(
                    &prompt,
                    ChatOptions {
                        max_new_tokens: args.max_new_tokens,
                        temperature: args.temperature,
                        top_p: args.top_p,
                        seed: args.seed,
                    },
                )?
            );
        }
        Command::Model { command } => handle_model(&root, command)?,
        Command::Capsule { command } => handle_capsule(&root, command)?,
        Command::Session { command } => handle_session(&root, command)?,
        Command::Vault { command } => handle_vault(&root, command)?,
        Command::Tools { command } => handle_tools(&root, command)?,
        Command::Skill { command } => handle_skill(&root, command)?,
        Command::Agent { command } => handle_agent(&root, command)?,
    }
    Ok(())
}

fn handle_model(root: &PathBuf, command: ModelCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        ModelCommand::Doctor => {
            let report = runtime.model_doctor()?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.healthy() {
                std::process::exit(2);
            }
        }
        ModelCommand::Verify => {
            let report = runtime.verify_model_assets()?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.ok() {
                std::process::exit(2);
            }
        }
        ModelCommand::Manifest { revision, files } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.write_model_manifest(&revision, &files)?)?
            );
        }
    }
    Ok(())
}

fn handle_capsule(root: &PathBuf, command: CapsuleCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        CapsuleCommand::Doctor => {
            let report = runtime.capsule_doctor()?;
            println!("{}", serde_json::to_string_pretty(&report)?);
            if !report.ok {
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

fn handle_session(root: &PathBuf, command: SessionCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        SessionCommand::Create { name } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.create_session(&name)?)?
            );
        }
        SessionCommand::List => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.list_sessions()?)?
            );
        }
        SessionCommand::Switch { id } => {
            runtime.switch_session(&id)?;
            println!("{}", serde_json::to_string_pretty(&json!({"active": id}))?);
        }
        SessionCommand::Snapshot { name } => {
            println!("{}", runtime.snapshot_active_session(&name)?.display());
        }
        SessionCommand::Restore { name } => {
            runtime.restore_active_session(&name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"restored": name}))?
            );
        }
        SessionCommand::Destroy { id } => {
            runtime.destroy_session(&id)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"destroyed": id}))?
            );
        }
    }
    Ok(())
}

fn handle_vault(root: &PathBuf, command: VaultCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        VaultCommand::Ingest { path } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.vault_ingest(path)?)?
            );
        }
        VaultCommand::Search { query, limit } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.vault_search(&query, limit)?)?
            );
        }
        VaultCommand::Lint => {
            let lint = runtime.vault_lint()?;
            println!("{}", serde_json::to_string_pretty(&lint)?);
            if !lint.is_clean() {
                std::process::exit(2);
            }
        }
    }
    Ok(())
}

fn handle_tools(root: &PathBuf, command: ToolsCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        ToolsCommand::List => println!("{}", serde_json::to_string_pretty(&runtime.list_tools()?)?),
        ToolsCommand::Register { manifest } => {
            runtime.register_tool_file(manifest)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({"registered": true}))?
            );
        }
        ToolsCommand::Call { id, input } => {
            let input = serde_json::from_str(&input).context("--input must be valid JSON")?;
            println!(
                "{}",
                serde_json::to_string_pretty(&runtime.call_tool(&id, input)?)?
            );
        }
        ToolsCommand::RegisterMcp { name, command } => {
            println!("{}", runtime.register_mcp(&name, &command)?.display());
        }
    }
    Ok(())
}

fn handle_skill(root: &PathBuf, command: SkillCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        SkillCommand::Create {
            id,
            name,
            description,
            body,
            global,
        } => println!(
            "{}",
            runtime
                .create_skill(&id, &name, &description, &body, global)?
                .display()
        ),
        SkillCommand::List => {
            println!("{}", serde_json::to_string_pretty(&runtime.list_skills()?)?)
        }
        SkillCommand::Delete { id } => {
            runtime.delete_skill(&id)?;
            println!("{}", serde_json::to_string_pretty(&json!({"deleted": id}))?);
        }
    }
    Ok(())
}

fn handle_agent(root: &PathBuf, command: AgentCommand) -> Result<()> {
    let runtime = KoaRuntime::open(root)?;
    match command {
        AgentCommand::Spawn {
            id,
            role,
            goal,
            output_contract,
            global,
        } => println!(
            "{}",
            runtime
                .spawn_agent(&id, &role, &goal, &output_contract, global)?
                .display()
        ),
        AgentCommand::List => {
            println!("{}", serde_json::to_string_pretty(&runtime.list_agents()?)?)
        }
        AgentCommand::Delete { id } => {
            runtime.delete_agent(&id)?;
            println!("{}", serde_json::to_string_pretty(&json!({"deleted": id}))?);
        }
    }
    Ok(())
}
