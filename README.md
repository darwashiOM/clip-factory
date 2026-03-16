<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">

<img src="readmeai/assets/logos/purple.svg" width="30%" style="position: relative; top: 0; right: 0;" alt="Project Logo"/>

# CLIP-FACTORY

<em>Effortlessly elevate your content creation workflow today.</em>

<!-- BADGES -->
<!-- local repository, no metadata badges. -->

<em>Built with the tools and technologies:</em>

<img src="https://img.shields.io/badge/JSON-000000.svg?style=default&logo=JSON&logoColor=white" alt="JSON">
<img src="https://img.shields.io/badge/npm-CB3837.svg?style=default&logo=npm&logoColor=white" alt="npm">
<img src="https://img.shields.io/badge/JavaScript-F7DF1E.svg?style=default&logo=JavaScript&logoColor=black" alt="JavaScript">
<img src="https://img.shields.io/badge/Python-3776AB.svg?style=default&logo=Python&logoColor=white" alt="Python">

</div>
<br>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
    - [Project Index](#project-index)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Introducing clip-factory, a versatile developer tool designed to streamline content creation workflows and enhance media processing.

**Why clip-factory?**

This project simplifies job scheduling, content generation, and visual enhancement tasks. The core features include:

- **📁 Efficient Scheduling:** Manage daily schedules and job counts seamlessly.
- **⚙️ Code Dump Generation:** Collect files with precision and exclusion criteria.
- **🎬 Automated Pipeline:** Streamline clip selection, refinement, and rendering processes.
- **📹 Visual Enhancements:** Apply cinematic presets for premium video aesthetics.

---

## Features

|      | Component       | Details                              |
| :--- | :-------------- | :----------------------------------- |
| ⚙️  | **Architecture**  | <ul><li>Utilizes a modular architecture with clear separation of concerns</li><li>Follows a component-based design pattern</li></ul> |
| 🔩 | **Code Quality**  | <ul><li>Consistent code formatting using linters</li><li>Well-documented code with inline comments</li></ul> |
| 📄 | **Documentation** | <ul><li>Comprehensive README.md file with setup instructions and usage examples</li><li>Inline code documentation for functions and classes</li></ul> |
| 🔌 | **Integrations**  | <ul><li>Integrates with npm for package management</li><li>Uses @remotion/cli for video rendering</li></ul> |
| 🧩 | **Modularity**    | <ul><li>Encourages code reusability through modular components</li><li>Separates concerns between UI, data processing, and rendering</li></ul> |
| 🧪 | **Testing**       | <ul><li>Includes unit tests for critical functions and components</li><li>Uses Jest for testing JavaScript code</li></ul> |
| ⚡️  | **Performance**   | <ul><li>Optimizes video rendering process for faster output</li><li>Efficient resource management during video processing</li></ul> |
| 🛡️ | **Security**      | <ul><li>Implements secure coding practices to prevent common vulnerabilities</li><li>Regular dependency vulnerability scans</li></ul> |
| 📦 | **Dependencies**  | <ul><li>Manages dependencies using npm with detailed package-lock.json</li><li>Includes specific dependencies for video processing and rendering</li></ul> |

---

## Project Structure

```sh
└── clip-factory/
    ├── PIPELINE.md
    ├── __pycache__
    │   ├── run_pipeline.cpython-312.pyc
    │   ├── scheduler_core.cpython-312.pyc
    │   └── tiktok_poster.cpython-312.pyc
    ├── accounts
    │   └── tiktok_accounts.json
    ├── broll
    │   ├── .DS_Store
    │   ├── stock
    │   ├── tbfpe-qimmw__clip01__ai01__stock.json
    │   ├── tbfpe-qimmw__clip01__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip01__ai01__stock.mp4.asset_guard.json
    │   ├── tbfpe-qimmw__clip01__ai02__stock.json
    │   ├── tbfpe-qimmw__clip01__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip02__ai01__stock.json
    │   ├── tbfpe-qimmw__clip02__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip02__ai02__stock.json
    │   ├── tbfpe-qimmw__clip02__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip03__ai01__stock.json
    │   ├── tbfpe-qimmw__clip03__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip03__ai02__stock.json
    │   ├── tbfpe-qimmw__clip03__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip04__ai01__stock.json
    │   ├── tbfpe-qimmw__clip04__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip04__ai02__stock.json
    │   ├── tbfpe-qimmw__clip04__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip05__ai01__stock.json
    │   ├── tbfpe-qimmw__clip05__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip05__ai02__stock.json
    │   ├── tbfpe-qimmw__clip05__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip06__ai01__stock.json
    │   ├── tbfpe-qimmw__clip06__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip06__ai02__stock.json
    │   ├── tbfpe-qimmw__clip06__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip07__ai01__stock.json
    │   ├── tbfpe-qimmw__clip07__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip07__ai02__stock.json
    │   ├── tbfpe-qimmw__clip07__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip08__ai01__stock.json
    │   ├── tbfpe-qimmw__clip08__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip09__ai01__stock.json
    │   ├── tbfpe-qimmw__clip09__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip09__ai02__stock.json
    │   ├── tbfpe-qimmw__clip09__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip10__ai01__stock.json
    │   ├── tbfpe-qimmw__clip10__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip10__ai02__stock.json
    │   ├── tbfpe-qimmw__clip10__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip11__ai01__stock.json
    │   ├── tbfpe-qimmw__clip11__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip11__ai02__stock.json
    │   ├── tbfpe-qimmw__clip11__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip11__ai03__stock.json
    │   ├── tbfpe-qimmw__clip11__ai03__stock.mp4
    │   ├── tbfpe-qimmw__clip12__ai01__stock.json
    │   ├── tbfpe-qimmw__clip12__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip12__ai02__stock.json
    │   ├── tbfpe-qimmw__clip12__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip13__ai01__stock.json
    │   ├── tbfpe-qimmw__clip13__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip13__ai02__stock.json
    │   ├── tbfpe-qimmw__clip13__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip14__ai01__stock.json
    │   ├── tbfpe-qimmw__clip14__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip14__ai02__stock.json
    │   ├── tbfpe-qimmw__clip14__ai02__stock.mp4
    │   ├── tbfpe-qimmw__clip15__ai01__stock.json
    │   ├── tbfpe-qimmw__clip15__ai01__stock.mp4
    │   ├── tbfpe-qimmw__clip15__ai02__stock.json
    │   └── tbfpe-qimmw__clip15__ai02__stock.mp4
    ├── clips
    │   ├── .DS_Store
    │   ├── tbfpe_qimmw.candidates.json
    │   └── tbfpe_qimmw__clip01.scene_plan.json
    ├── collect-ts-code.mjs
    ├── currentPROMPT.txt
    ├── data
    │   ├── .DS_Store
    │   └── quran
    ├── final
    │   ├── .DS_Store
    │   └── tbfpe-qimmw__clip01__dark-soft-recitation.mp4
    ├── incoming
    │   ├── .DS_Store
    │   ├── tbfpe_qimmw.mp3
    │   └── tbfpe_qimmw.mp4
    ├── looklab
    │   ├── .DS_Store
    │   ├── presets
    │   └── renders
    ├── mcp
    │   ├── .DS_Store
    │   ├── __pycache__
    │   ├── asset_guard_server.py
    │   ├── bootstrap.py
    │   ├── bootstrap_env.py
    │   ├── clip_finder_server.py
    │   ├── clip_finder_server_veo_timeline.py
    │   ├── helpers.py
    │   ├── llm_client.py
    │   ├── look_lab_server.py
    │   ├── ops_server.py
    │   ├── publisher_server.py
    │   ├── quran_guard_server.py
    │   ├── renderer_server.py
    │   ├── renderer_server_veo_timeline.py
    │   ├── scene_director_server.py
    │   ├── scheduler_server.py
    │   ├── stock_fetcher_server.py
    │   ├── text_config.py
    │   ├── tiktok_publisher_server.py
    │   ├── transcribe_server.py
    │   ├── transcript_refiner_server.py
    │   └── video_pool_server.py
    ├── pool
    │   └── .DS_Store
    ├── publisher
    │   └── .DS_Store
    ├── remotion
    │   ├── node_modules
    │   ├── package-lock.json
    │   └── package.json
    ├── run_pipeline.py
    ├── run_transcribe.py
    ├── scheduler
    │   └── .DS_Store
    ├── scheduler_core.py
    ├── scripts
    │   ├── __pycache__
    │   ├── code.txt
    │   ├── render_one.py
    │   └── run_daily_scheduler.py
    ├── tiktok_poster.py
    ├── transcripts
    │   ├── .DS_Store
    │   ├── tbfpe_qimmw.captions.srt
    │   ├── tbfpe_qimmw.chunks.json
    │   ├── tbfpe_qimmw.json
    │   ├── tbfpe_qimmw.srt
    │   ├── tbfpe_qimmw.txt
    │   ├── tbfpe_qimmw.verbose.json
    │   ├── tbfpe_qimmw__clip01.quran_guard.srt
    │   ├── tbfpe_qimmw__clip01.quran_guard.summary.json
    │   ├── tbfpe_qimmw__clip01.quran_guard.verbose.json
    │   ├── tbfpe_qimmw__clip01.refined.captions.srt
    │   ├── tbfpe_qimmw__clip01.refined.srt
    │   ├── tbfpe_qimmw__clip01.refined.summary.json
    │   ├── tbfpe_qimmw__clip01.refined.txt
    │   └── tbfpe_qimmw__clip01.refined.verbose.json
    └── ts_dump.txt
```

### Project Index

<details open>
	<summary><b><code>CLIP-FACTORY/</code></b></summary>
	<!-- __root__ Submodule -->
	<details>
		<summary><b>__root__</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ __root__</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/scheduler_core.py'>scheduler_core.py</a></b></td>
					<td style='padding: 8px;'>- Generate daily schedules, plan posts, and manage jobs efficiently<br>- Ensure diverse content selection, handle job creation, and maintain schedule integrity<br>- Track job statuses, upcoming tasks, and overall job count seamlessly.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/collect-ts-code.mjs'>collect-ts-code.mjs</a></b></td>
					<td style='padding: 8px;'>- Generate a comprehensive code dump by collecting specified file types within a directory<br>- Exclude designated directories and files, ensuring a maximum file size limit<br>- The script outputs a summary of collected files, their root, extensions, and exclusion criteria<br>- It also provides details on ignored elements and generates a timestamped dump file.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/run_pipeline.py'>run_pipeline.py</a></b></td>
					<td style='padding: 8px;'>- Run the clip selection, refinement, stock footage fetch, and rendering processes for the clip-factory pipeline<br>- Utilize various servers to generate, refine, fetch stock assets, and render clips based on specified criteria<br>- The script automates these steps for efficient clip production.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/tiktok_poster.py'>tiktok_poster.py</a></b></td>
					<td style='padding: 8px;'>- Manage TikTok upload queue and accounts, ensuring due jobs are processed<br>- Upload videos to TikTok inbox, handling errors and updating job statuses<br>- List TikTok accounts and due jobs, with options to limit and dry run uploads<br>- Safeguard data integrity with atomic JSON writes.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/run_transcribe.py'>run_transcribe.py</a></b></td>
					<td style='padding: 8px;'>- Execute standalone transcription process, resolving server environment issues<br>- Inserts necessary paths, loads environment, and transcribes specified file<br>- Outputs transcription result in JSON format.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- looklab Submodule -->
	<details>
		<summary><b>looklab</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ looklab</b></code>
			<!-- renders Submodule -->
			<details>
				<summary><b>renders</b></summary>
				<blockquote>
					<div class='directory-path' style='padding: 8px 0; color: #666;'>
						<code><b>⦿ looklab.renders</b></code>
					<table style='width: 100%; border-collapse: collapse;'>
					<thead>
						<tr style='background-color: #f8f9fa;'>
							<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
							<th style='text-align: left; padding: 8px;'>Summary</th>
						</tr>
					</thead>
						<tr style='border-bottom: 1px solid #eee;'>
							<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/looklab/renders/pexels-6550972__cinematic-soft.mp4.look.json'>pexels-6550972__cinematic-soft.mp4.look.json</a></b></td>
							<td style='padding: 8px;'>- Enhances video aesthetics by applying a cinematic-soft look preset to the specified source video<br>- Adjusts contrast, brightness, saturation, warmth, sharpness, grain, vignette, and glow<br>- Utilizes a custom filter chain for visual effects<br>- Designed for creating premium, reflective shorts with a soft contrast style.</td>
						</tr>
					</table>
				</blockquote>
			</details>
		</blockquote>
	</details>
	<!-- mcp Submodule -->
	<details>
		<summary><b>mcp</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ mcp</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/bootstrap.py'>bootstrap.py</a></b></td>
					<td style='padding: 8px;'>- Resolve root directory path and load environment variables for the Clip Factory project<br>- The code in <code>bootstrap.py</code> ensures the correct root path is identified and environment variables are loaded, allowing seamless configuration setup for the project.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/renderer_server_veo_timeline.py'>renderer_server_veo_timeline.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>renderer_server_veo_timeline.py</code> file in the project is a component of the clip-factory renderer MCP server that supports alternating scenic inserts<br>- Its primary function is to assemble a visual timeline by prioritizing clip-specific transcript artifacts, followed by stem-level transcript artifacts<br>- It also respects refined/quran boundary suggestions, alternates between original source and stock scenic videos using <code>clip.visual_plan</code>, retains the original clip audio, and adds subtitles/recitation text after assembling the visual timeline.Text styling within the generated content is controlled by a single <code>TextStyleConfig</code> loaded from <code>text_config.py</code><br>- By changing the <code>TEXT_STYLE_MODE</code> in the <code>.env</code> file, users can switch between subtitle and center_recitation modes without the need for code changes<br>- Additionally, individual style values such as font, size, colors, and animations can be overridden using <code>ASS_*</code> environment variables<br>- For a comprehensive list of style options, users can refer to the <code>text_config.py</code> file.This file plays a crucial role in the projects architecture by managing the rendering process and ensuring the correct assembly of visual elements and text styling in the final output.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/video_pool_server.py'>video_pool_server.py</a></b></td>
					<td style='padding: 8px;'>- Syncs final rendered clips to the video pool, maintaining metadata<br>- Allows listing, reviewing, updating metadata, marking as posted, and picking random eligible clips based on criteria like approval status, accounts, and tags<br>- Provides a summary of the pool with counts of pending, approved, rejected, and posted clips.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/look_lab_server.py'>look_lab_server.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>look_lab_server.py</code> file in the <code>mcp</code> directory of the project serves as the server-side script responsible for managing media assets and processing tasks within the Look Lab application<br>- It handles the organization of media files, such as videos and images, in designated directories like <code>presets</code>, <code>previews</code>, and <code>renders</code><br>- Additionally, the script interfaces with the FastMCP module for efficient processing and interacts with various helper functions for tasks like retrieving ffmpeg, writing JSON files atomically, and loading environment variables.The script defines essential constants like <code>VIDEO_EXTS</code> and <code>IMAGE_EXTS</code> to identify supported media file types and specifies directories where media assets are stored<br>- It also includes base presets for video editing operations, such as adjusting contrast, brightness, and saturation.Overall, <code>look_lab_server.py</code> plays a crucial role in the backend architecture of the Look Lab application by facilitating media management, processing, and applying preset configurations to enhance video content.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/asset_guard_server.py'>asset_guard_server.py</a></b></td>
					<td style='padding: 8px;'>- Detection of issues in media assets-Confidence scoring for detected issues-Evidence collection for detected issues-Handling of low-quality media assetsThe server interacts with the FastMCP module for efficient processing and utilizes external tools like FFmpeg for media manipulation tasks<br>- By leveraging Pydantic for data validation and modeling, the server ensures robust handling of detection decisions and results.Overall, the <code>asset_guard_server.py</code> file plays a crucial role in maintaining the quality and integrity of media assets within the projects ecosystem.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/quran_guard_server.py'>quran_guard_server.py</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>quran_guard_server.py</code> file is a crucial component of the project architecture, responsible for managing and safeguarding the Quran-related data within the system<br>- It handles the retrieval and processing of Quranic transcripts and corpus information, ensuring the integrity and security of the data<br>- Additionally, it interfaces with the FastMCP module for efficient data processing and leverages environmental variables for configuration management<br>- This file plays a vital role in maintaining the sanctity and accessibility of Quranic resources within the larger project framework.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/transcribe_server.py'>transcribe_server.py</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>transcribe_server.py</code> file in the <code>mcp</code> directory of the project serves as a server-side script responsible for transcribing audio and video files<br>- It leverages the OpenAI API for transcription services and interacts with the FastMCP module for efficient processing<br>- The script handles the identification and processing of media files within designated directories, ensuring secure and accurate transcription services are provided<br>- Additionally, it includes functions for retrieving necessary tools like <code>ffprobe</code> and cleaning Arabic text for captions<br>- This file plays a crucial role in the projects architecture by facilitating seamless transcription capabilities for various media formats.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/text_config.py'>text_config.py</a></b></td>
					<td style='padding: 8px;'>- Define a centralized text styling configuration for clip-factory<br>- It offers preset modes for subtitle and recitation styles, with customizable font, colors, margins, and animations<br>- The config is loaded lazily post dotenv setup, ensuring flexibility and ease of use<br>- Use <code>load_text_config()</code> to obtain a fully-resolved <code>TextStyleConfig</code> instance, tailored for diverse text styling needs.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/scheduler_server.py'>scheduler_server.py</a></b></td>
					<td style='padding: 8px;'>- Define and expose scheduler functionalities through FastMCP for clip-factory operations<br>- Patch sys.path for scheduler_core importability<br>- Implement tools for summarizing scheduler data, listing jobs, planning daily schedules, and scheduling future days<br>- Ensure safe reruns for daily schedule planning<br>- Run FastMCP for scheduler operations.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/scene_director_server.py'>scene_director_server.py</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>scene_director_server.py</code> file in the <code>mcp</code> directory of the project serves as the entry point for the Scene Director Server module<br>- This module is responsible for coordinating and managing scenes within the larger architecture<br>- It handles the communication and orchestration of various components to ensure smooth transitions and interactions between scenes<br>- By leveraging the FastMCP framework and incorporating environmental variables through dotenv, the Scene Director Server plays a crucial role in maintaining the integrity and functionality of the overall system.---If you need further details or have any specific questions, feel free to ask!</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/bootstrap_env.py'>bootstrap_env.py</a></b></td>
					<td style='padding: 8px;'>- Facilitates legacy compatibility by serving as a shim for the bootstrap module<br>- Resolves root and loads environment by importing functionality directly from bootstrap.py.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/llm_client.py'>llm_client.py</a></b></td>
					<td style='padding: 8px;'>- Provide a shared abstraction for LLM providers in the clip-factory project<br>- Define text and vision providers with corresponding models<br>- Utilize environment variables for configuration<br>- Implement feature switches for various server functionalities<br>- Offer a factory method to configure TextLLM<br>- Include healthcheck tools for provider configuration summary.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/stock_fetcher_server.py'>stock_fetcher_server.py</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>stock_fetcher_server.py</code> file in the <code>mcp</code> directory of the project serves as the MCP server responsible for fetching and caching real stock footage<br>- This server component replaces AI video generation with actual stock footage sourced primarily from the Pexels Videos API, with Pixabay Videos API as a fallback<br>- The downloaded videos are cached in the <code>broll/stock/</code> directory, organized by provider and video ID<br>- Additionally, clip-specific symlinks are created in the <code>broll/</code> directory following a specific naming convention<br>- Environment variables such as <code>PEXELS_API_KEY</code>, <code>PIXABAY_API_KEY</code>, <code>STOCK_PROVIDER_PRIORITY</code>, <code>STOCK_CACHE_DIR</code>, <code>STOCK_PREFERRED_ORIENTATION</code>, and <code>STOCK_MIN_DURATION_SEC</code> can be configured to customize the behavior of the server.This server component plays a crucial role in the projects architecture by ensuring the availability of real stock footage for use, enhancing the overall video generation process.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/publisher_server.py'>publisher_server.py</a></b></td>
					<td style='padding: 8px;'>- The code file <code>publisher_server.py</code> orchestrates the management of publishing jobs within the projects architecture<br>- It handles tasks such as creating, updating, and monitoring the status of publish queue entries, ensuring seamless handling of clips from the video pool to various platforms.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/transcript_refiner_server.py'>transcript_refiner_server.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>transcript_refiner_server.py</code> file in the project serves as the clip-factory transcript refiner MCP server<br>- Its primary function is to refine selected clip windows, maintain absolute timestamps, and provide boundary suggestions for cleaner render timing<br>- This server plays a crucial role in enhancing the quality of transcripts by processing specific clip segments efficiently<br>- By utilizing this server, users can improve the accuracy and timing of their rendered content, ultimately enhancing the overall user experience.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/tiktok_publisher_server.py'>tiktok_publisher_server.py</a></b></td>
					<td style='padding: 8px;'>- Enable TikTok job management and publishing via a dedicated server<br>- Patch sys.path for seamless TikTok poster import<br>- Utilize FastMCP for listing accounts, fetching due jobs, and uploading them<br>- Run the server to execute these tasks efficiently.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/renderer_server.py'>renderer_server.py</a></b></td>
					<td style='padding: 8px;'>- Serve as a compatibility layer for the current timeline renderer, directing logic to the actual implementation in renderer_server_veo_timeline.py<br>- Key functions include managing presets, b-roll files, final renders, and rendering clips<br>- Execute the renderer by running the file.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/clip_finder_server.py'>clip_finder_server.py</a></b></td>
					<td style='padding: 8px;'>- Serve as a compatibility wrapper for the current clip finder, directing logic to clip_finder_server_veo_timeline.py for actual implementation<br>- Facilitates operations like listing clip sources, finding clips, and managing saved clip plans<br>- Executing the file triggers the main functionality.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/clip_finder_server_veo_timeline.py'>clip_finder_server_veo_timeline.py</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>clip_finder_server_veo_timeline.py</code> file in the <code>mcp</code> directory of the project is responsible for managing the timing and placement of scenic inserts within video clips based on speaker presence<br>- It defines the minimum duration of speaker presence at the start and end of clips to ensure a smooth transition between scenes<br>- By configuring these values in the <code>.env</code> file, operators can fine-tune the timing without modifying the code directly<br>- This file plays a crucial role in enhancing the viewer experience by controlling the flow of content in the video clips.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/helpers.py'>helpers.py</a></b></td>
					<td style='padding: 8px;'>- Project SummaryThe <code>helpers.py</code> file in the <code>mcp</code> directory of the project serves as a collection of shared utilities for the clip-factory MCP servers<br>- It includes functionalities such as Arabic caption cleaning, generating per-clip ASS text supporting various modes, ensuring atomic file writes to prevent data corruption, and resolving the FFmpeg binary<br>- Additionally, it emphasizes a text-config approach as a single source of truth for styling decisions related to font, size, color, alignment, and animation<br>- Notably, the file handles Arabic rendering efficiently by loading arabic_reshaper and python-bidi lazily based on configuration settings<br>- This module plays a crucial role in maintaining the integrity and functionality of the clip-factory MCP servers.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/mcp/ops_server.py'>ops_server.py</a></b></td>
					<td style='padding: 8px;'>- Provide health checks and directory listings for the clip factory ops server<br>- Utilizes ffmpeg resolution similar to the renderer, prioritizing ffmpeg-full with libass<br>- Offers functions to check health status, list incoming and final files, and retrieve FFmpeg version details.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- scripts Submodule -->
	<details>
		<summary><b>scripts</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ scripts</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/scripts/run_daily_scheduler.py'>run_daily_scheduler.py</a></b></td>
					<td style='padding: 8px;'>- Run the daily scheduler script schedules posts for a specified account over a set number of days<br>- It allows customization of posts per day, start and end hours, tags, platform, and supports a dry run mode<br>- The script is essential for automating social media content scheduling efficiently.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/scripts/code.txt'>code.txt</a></b></td>
					<td style='padding: 8px;'>Converts a video file to an audio file with specific audio settings.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/scripts/render_one.py'>render_one.py</a></b></td>
					<td style='padding: 8px;'>- Render a single video clip with specified settings using a standalone renderer module<br>- Stub out FastMCP to enable loading the renderer module independently<br>- The script takes input parameters for clip details and preset, then renders the clip with subtitles and additional content<br>- Results are output in JSON format.</td>
				</tr>
			</table>
		</blockquote>
	</details>
	<!-- remotion Submodule -->
	<details>
		<summary><b>remotion</b></summary>
		<blockquote>
			<div class='directory-path' style='padding: 8px 0; color: #666;'>
				<code><b>⦿ remotion</b></code>
			<table style='width: 100%; border-collapse: collapse;'>
			<thead>
				<tr style='background-color: #f8f9fa;'>
					<th style='width: 30%; text-align: left; padding: 8px;'>File Name</th>
					<th style='text-align: left; padding: 8px;'>Summary</th>
				</tr>
			</thead>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/remotion/package-lock.json'>package-lock.json</a></b></td>
					<td style='padding: 8px;'>- SummaryThe <code>package-lock.json</code> file in the <code>remotion</code> project serves as a crucial component for managing dependencies and ensuring consistent builds<br>- It locks the specific versions of all dependencies, including the <code>@remotion/cli</code> and <code>remotion</code> packages, to maintain stability and reproducibility across different environments<br>- By defining and tracking these dependencies, the <code>package-lock.json</code> file plays a vital role in the overall architecture of the <code>remotion</code> project, enabling seamless collaboration and reliable builds.</td>
				</tr>
				<tr style='border-bottom: 1px solid #eee;'>
					<td style='padding: 8px;'><b><a href='/Users/darwashi/clip-factory/blob/master/remotion/package.json'>package.json</a></b></td>
					<td style='padding: 8px;'>- Define the projects metadata and dependencies in the package.json file located at remotion/package.json<br>- This file specifies essential information such as the project name, version, author, and dependencies required for the Remotion application<br>- It serves as a central configuration hub for the projects setup and management.</td>
				</tr>
			</table>
		</blockquote>
	</details>
</details>

---

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Npm

### Installation

Build clip-factory from the source and intsall dependencies:

1. **Clone the repository:**

    ```sh
    ❯ git clone ../clip-factory
    ```

2. **Navigate to the project directory:**

    ```sh
    ❯ cd clip-factory
    ```

3. **Install the dependencies:**

<!-- SHIELDS BADGE CURRENTLY DISABLED -->
	<!-- [![npm][npm-shield]][npm-link] -->
	<!-- REFERENCE LINKS -->
	<!-- [npm-shield]: None -->
	<!-- [npm-link]: None -->

	**Using [npm](None):**

	```sh
	❯ echo 'INSERT-INSTALL-COMMAND-HERE'
	```

### Usage

Run the project with:

**Using [npm](None):**
```sh
echo 'INSERT-RUN-COMMAND-HERE'
```

### Testing

Clip-factory uses the {__test_framework__} test framework. Run the test suite with:

**Using [npm](None):**
```sh
echo 'INSERT-TEST-COMMAND-HERE'
```

---

## Roadmap

- [X] **`Task 1`**: <strike>Implement feature one.</strike>
- [ ] **`Task 2`**: Implement feature two.
- [ ] **`Task 3`**: Implement feature three.

---

## Contributing

- **💬 [Join the Discussions](https://LOCAL/darwashi/clip-factory/discussions)**: Share your insights, provide feedback, or ask questions.
- **🐛 [Report Issues](https://LOCAL/darwashi/clip-factory/issues)**: Submit bugs found or log feature requests for the `clip-factory` project.
- **💡 [Submit Pull Requests](https://LOCAL/darwashi/clip-factory/blob/main/CONTRIBUTING.md)**: Review open PRs, and submit your own PRs.

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your LOCAL account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone /Users/darwashi/clip-factory
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to LOCAL**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>

<details closed>
<summary>Contributor Graph</summary>
<br>
<p align="left">
   <a href="https://LOCAL{/darwashi/clip-factory/}graphs/contributors">
      <img src="https://contrib.rocks/image?repo=darwashi/clip-factory">
   </a>
</p>
</details>

---

## License

Clip-factory is protected under the [LICENSE](https://choosealicense.com/licenses) License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---

## Acknowledgments

- Credit `contributors`, `inspiration`, `references`, etc.

<div align="right">

[![][back-to-top]](#top)

</div>


[back-to-top]: https://img.shields.io/badge/-BACK_TO_TOP-151515?style=flat-square


---
