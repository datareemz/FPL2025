variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
}

variable "region" {
  type        = string
  description = "The GCP region (e.g., 'us-central1')."
  default     = "us-central1"
}

variable "repository_id" {
  type        = string
  description = "The name for your Artifact Registry Docker repository."
  default     = "my-python-apps"
}

variable "github_repo" {
  type        = string
  description = "Your GitHub repository in 'owner/repo_name' format (e.g., 'my-org/my-cool-app')."
}
