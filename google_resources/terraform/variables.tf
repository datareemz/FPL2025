variable "project_id" {
  type        = string
  description = "Your Google Cloud project ID."
  default     = "default"
}

variable "region" {
  type        = string
  description = "The GCP region where resources will be created"
  default     = "us-central1"
}

variable "repository_id" {
  type        = string
  description = "The name for your Artifact Registry Docker repository."
  default     = "fpl-2025"
}
