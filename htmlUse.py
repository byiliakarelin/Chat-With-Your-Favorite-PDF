# HTML & CSS Codes

# Updated CSS with @import for css.gg camera and album icons
css = '''
<style>
@import url('https://unpkg.com/css.gg@2.0.0/icons/css/ghost-character.css');
@import url('https://unpkg.com/css.gg@2.0.0/icons/css/profile.css');

.chat-message {
    padding: 1.5rem;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    display: flex;
    background-color: #ffffff; /* Plain white background */
    border: 2px solid #d1d5db; /* Light gray border */
}
.chat-message.user {
    border-left: 4px solid #000000; /* Blue border on the left for user messages */
}
.chat-message.bot {
    border-left: 4px solid #3e52d4; /* My branding color */
}
.chat-message .avatar {
    width: 20%;
    display: flex;
    justify-content: center;
    align-items: center;
}
.chat-message .message {
    width: 80%;
    padding: 0 1.5rem;
    color: #000;
}
</style>
'''

# Updated bot_template with css.gg camera icon
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <i class="gg-ghost-character"></i>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''

# Updated user_template with css.gg album icon
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <i class="gg-profile"></i>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
